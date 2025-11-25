import json
import logging
import threading
import tempfile
from pathlib import Path
import re

import tkinter as tk
from PIL import Image
import torch
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
from pynput.keyboard import GlobalHotKeys
import pyperclip

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
    """Load HunyuanOCR model."""
    model_name = cfg["model"]
    log.info(f"Loading model: {model_name}")
    
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_name,
        attn_implementation="eager",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if torch.cuda.is_available():
        log.info("Using CUDA with bfloat16")
    else:
        log.info("Using CPU with float32")
    
    return processor, model


def copy_to_clipboard(text: str) -> None:
    pyperclip.copy(text)


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


def transcribe_image(img: Image.Image, processor: AutoProcessor, model: HunYuanVLForConditionalGeneration, cfg: dict) -> str:
    """Transcribe image to markdown using direct model inference."""
    max_side = cfg["max_side"]
    img = preprocess_image(img, max_side)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_img_path = Path(temp_dir) / "input.png"
            img.save(temp_img_path, "PNG")
            
            prompt_text = get_prompt(cfg)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(temp_img_path)},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            texts = [
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ]
            
            inputs = processor(
                text=texts,
                images=[img],
                padding=True,
                return_tensors="pt",
            )
            
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=16384, do_sample=False)
                
            if "input_ids" in inputs:
                input_ids = inputs.input_ids
            else:
                input_ids = inputs.inputs
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            result = output_texts[0] if output_texts else ""

            if cfg["use_dollars_for_math"]:
                result = convert_math_delimiters(result)
            return result
            
    except Exception as e:
        log.error("OCR inference failed: %s", e)
        return f"Error during OCR: {str(e)}"


class OCRHotkeyApp:
    """Main application handling hotkeys and OCR processing."""
    
    def __init__(self, processor: AutoProcessor, model: HunYuanVLForConditionalGeneration, config: dict):
        self.processor = processor
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
                        img, self.processor, self.model, self.config)
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
    processor, model = load_ocr_model(cfg)
    app = OCRHotkeyApp(processor, model, cfg)
    app.run()


if __name__ == "__main__":
    main()
