import json
import logging
import threading
import tempfile
from pathlib import Path
import re

import tkinter as tk
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from pynput.keyboard import GlobalHotKeys

from region_selector import RegionSelector


logging.basicConfig(
    level=logging.INFO,
    format="[markdowner] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(img: Image.Image, max_side: int) -> Image.Image:
    """Resize image if too large."""
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    
    if max(img.size) > max_side > 0:
        scale = max_side / max(img.size)
        new_size = (
            max(1, int(img.width * scale)),
            max(1, int(img.height * scale))
        )
        img = img.resize(new_size, Image.BICUBIC)
    
    return img


def load_ocr_model(cfg: dict):
    """Load DeepSeek-OCR model."""
    model_name = cfg["model"]
    log.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        use_safetensors=True
    )
    
    if torch.cuda.is_available():
        model = model.eval().cuda().to(torch.bfloat16)
        log.info("Using CUDA with bfloat16")
    else:
        model = model.eval().to(torch.float32)
        log.info("Using CPU with float32")
    
    return tokenizer, model


def copy_to_clipboard(text: str) -> None:
    """Copy text to clipboard using tkinter."""
    root = tk.Tk()
    root.withdraw()
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()
    root.destroy()


def convert_math_delimiters(text):
    """Convert LaTeX-style math delimiters to dollar signs with regex."""
    text = re.sub(r'\\\(\s*', '$', text)   # \( --> $
    text = re.sub(r'\s*\\\)', '$', text)   # (\ --> $
    text = re.sub(r'\\\[\s*', '$$', text)  # \[ --> $$
    text = re.sub(r'\s*\\\]', '$$', text)  # \] --> $$
    return text


def get_prompt(cfg: dict) -> str:
    with open(cfg["prompt_path"], "r", encoding="utf-8") as f:
        return f.read()


def transcribe_image(img: Image.Image, tokenizer: AutoTokenizer, model: AutoModel, cfg: dict) -> str:
    """Transcribe image to markdown using direct model inference."""
    max_side = cfg["max_side"]
    img = preprocess_image(img, max_side)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_img_path = Path(temp_dir) / "input.png"
            img.save(temp_img_path, "PNG")

            # Run OCR inference
            # model.infer always outputs None. Even when save = False.
            model.infer(
                tokenizer,
                prompt=f"<image>\n<|grounding|>{get_prompt(cfg)}",
                image_file=temp_img_path,
                output_path=temp_dir,
                base_size=cfg["base_size"],
                image_size=cfg["image_size"],
                crop_mode=cfg["crop_mode"],
                save_results=True,
                test_compress=cfg["test_compress"]
            )
            
            result = extract_markdown_from_temp(temp_dir)

            if cfg["use_dollars_for_math"]:
                result = convert_math_delimiters(result)
            return result
            
    except Exception as e:
        log.error("OCR inference failed: %s", e)
        return f"Error during OCR: {str(e)}"


def extract_markdown_from_temp(temp_dir: str) -> str:
    """Extract markdown text from temporary output directory."""
    try:
        temp_path = Path(temp_dir)
        
        # Look for any text files in the temp directory
        for ext in ('.mmd', '.txt', '.md', '.markdown'):
            for text_file in temp_path.rglob(f"*{ext}"):
                content = text_file.read_text(encoding="utf-8").strip()
                if content:
                    log.debug("Found markdown content: %.500s...", content)
                    return content
        
        # Check if there are any files at all in the temp directory
        all_files = list(temp_path.rglob("*"))
        if all_files:
            log.debug("Found files in temp dir: %s", [f.name for f in all_files])
        
    except Exception as e:
        log.error("Error extracting markdown from temp dir: %s", e)
    
    return "No OCR result obtained"


class OCRHotkeyApp:
    """Main application handling hotkeys and OCR processing."""
    
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModel, config: dict):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.listener = None
        self._busy = threading.Lock()

    def on_ocr(self):
        if self._busy.locked():
            log.info("Already processing a capture")
            return
        
        def process_ocr():
            with self._busy:
                try:
                    img = RegionSelector().select()
                    if img is None:
                        log.warning("Failed to capture image")
                        return
                    markdown = transcribe_image(
                        img, self.tokenizer, self.model, self.config)
                    copy_to_clipboard(markdown)
                    log.info("Copied Markdown to clipboard")
                
                except Exception as e:
                    log.exception("Transcription error: %s", e)
        
        threading.Thread(target=process_ocr, daemon=True).start()
    
    def on_quit(self):
        log.info("Exiting on hotkey")
        if self.listener:
            self.listener.stop()
    
    def run(self):
        log.info(f"Ready. Press {self.config["hotkey_ocr"]} to OCR; {self.config["hotkey_quit"]} to quit")
                
        hotkeys = {
            self.config["hotkey_ocr"]: self.on_ocr,
            self.config["hotkey_quit"]: self.on_quit,
        }
        
        with GlobalHotKeys(hotkeys) as listener:
            self.listener = listener
            listener.join()


def main():
    cfg = load_config("config.json")
    if cfg["debug"]:
        logging.getLogger().setLevel(logging.DEBUG)
    tokenizer, model = load_ocr_model(cfg)
    app = OCRHotkeyApp(tokenizer, model, cfg)
    app.run()


if __name__ == "__main__":
    main()
