import json
import logging
import threading
import tempfile
from pathlib import Path
from typing import Optional
import re

import tkinter as tk
from PIL import ImageGrab, Image
import torch
from transformers import AutoModel, AutoTokenizer


# Simplified logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[markdowner] %(levelname)s: %(message)s"
)
log = logging.getLogger("markdowner")


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        if config["debug"]:
            logging.getLogger().setLevel(logging.DEBUG)
        
        log.info("Hotkeys -> OCR: %s | Quit: %s", 
                config["hotkey_ocr"], 
                config["hotkey_quit"])
        return config
        
    except Exception as e:
        log.error("Failed to load config: %s", e)
        raise


def get_clipboard_image() -> Optional[Image.Image]:
    """Get image from clipboard."""
    try:
        root = tk.Tk()
        root.withdraw()
        return ImageGrab.grabclipboard()
    except Exception:
        return None


class RegionSelector:
    """Cross-platform region selector using Tkinter."""
    
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.result: Optional[Image.Image] = None
    
    def _on_press(self, event, bbox: dict, canvas: tk.Canvas):
        bbox["x0"], bbox["y0"] = event.x, event.y
        canvas.delete("rect")
    
    def _on_drag(self, event, bbox: dict, canvas: tk.Canvas):
        if bbox["x0"] is None or bbox["y0"] is None:
            return
        
        canvas.delete("rect")
        canvas.create_rectangle(
            bbox["x0"], bbox["y0"], event.x, event.y,
            outline="#00D1B2", width=2, tags="rect"
        )
    
    def _on_release(self, event, bbox: dict, root: tk.Tk):
        if bbox["x0"] is None or bbox["y0"] is None:
            root.quit()
            return
        
        x0, x1 = sorted([bbox["x0"], event.x])
        y0, y1 = sorted([bbox["y0"], event.y])
        
        try:
            self.result = ImageGrab.grab(bbox=(x0, y0, x1, y1))
        except Exception as e:
            log.debug("ImageGrab failed: %s", e)
        finally:
            root.quit()
    
    def select(self) -> Optional[Image.Image]:
        """Show region selector and return captured image."""
        root = tk.Tk()
        root.attributes("-fullscreen", True)
        root.attributes("-alpha", 0.15)
        root.configure(bg="black")
        root.attributes("-topmost", True)
        
        canvas = tk.Canvas(root, cursor="cross", highlightthickness=0, bg="black")
        canvas.pack(fill=tk.BOTH, expand=True)
        
        bbox = {"x0": None, "y0": None}
        
        # Bind events
        canvas.bind("<ButtonPress-1>", lambda e: self._on_press(e, bbox, canvas))
        canvas.bind("<B1-Motion>", lambda e: self._on_drag(e, bbox, canvas))
        canvas.bind("<ButtonRelease-1>", lambda e: self._on_release(e, bbox, root))
        root.bind("<Escape>", lambda e: root.quit())
        
        # Auto-cancel after timeout
        root.after(int(self.timeout * 1000), root.quit)
        
        try:
            root.mainloop()
        finally:
            try:
                root.destroy()
            except Exception:
                pass
        
        return self.result


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
    log.info("Loading DeepSeek-OCR model: %s", model_name)
    
    try:
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
        
    except Exception as e:
        log.error("Failed to load DeepSeek-OCR model: %s", e)
        raise


def copy_to_clipboard(text: str) -> None:
    """Copy text to clipboard using tkinter."""
    root = tk.Tk()
    root.withdraw()
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()
    root.destroy()



def convert_math_delimiters(text):
    """
    Convert LaTeX-style math delimiters to dollar signs with regex.
    Handles various whitespace patterns around delimiters.
    """
    # Replace \( with optional whitespace with $
    text = re.sub(r'\\\(\s*', '$', text)
    
    # Replace \) with optional whitespace with $
    text = re.sub(r'\s*\\\)', '$', text)
    
    # Replace \[ with optional whitespace with $$
    text = re.sub(r'\\\[\s*', '$$', text)
    
    # Replace \] with optional whitespace with $$
    text = re.sub(r'\s*\\\]', '$$', text)
    
    return text


def get_prompt(cfg: dict) -> str:
    with open(cfg["prompt_path"], "r", encoding="utf-8") as f:
        return f.read()


def transcribe_image_direct(img: Image.Image, tokenizer: AutoTokenizer, model: AutoModel, cfg: dict) -> str:
    """Transcribe image to markdown using direct model inference."""
    max_side = cfg["max_side"]
    img = preprocess_image(img, max_side)
    
    # Save to temporary file (required by DeepSeek-OCR infer method)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        img.save(temp_path, "PNG")
    
    try:
        # Use a temporary output directory that we'll clean up immediately
        with tempfile.TemporaryDirectory() as temp_output:
            # Run OCR inference
            # model.infer always outputs None... The same goes with the official example. Even when save
            model.infer(
                tokenizer,
                prompt=f"<image>\n<|grounding|>{get_prompt(cfg)}",
                image_file=temp_path,
                output_path=temp_output,
                base_size=cfg["base_size"],
                image_size=cfg["image_size"],
                crop_mode=cfg["crop_mode"],
                save_results=True,
                test_compress=cfg["test_compress"]
            )
            
            result = extract_markdown_from_temp(temp_output)

            if cfg["use_dollars_for_math"]:
                result = convert_math_delimiters(result)
            return result
        
    except Exception as e:
        log.error("OCR inference failed: %s", e)
        return f"Error during OCR: {str(e)}"
    finally:
        # Clean up temporary file
        try:
            Path(temp_path).unlink()
        except OSError:
            log.debug("Could not clean up temp file")


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


def capture_and_transcribe(tokenizer: AutoTokenizer, model: AutoModel, cfg: dict) -> Optional[str]:
    """Capture screenshot and transcribe to markdown."""
    log.info("Using built-in region selector")
    img = RegionSelector().select()
    
    if img is None:
        log.warning("Screenshot cancelled or unavailable")
        return None
    
    return transcribe_image_direct(img, tokenizer, model, cfg)


class OCRHotkeyApp:
    """Main application handling hotkeys and OCR processing."""
    
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModel, config: dict):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.listener = None
        self._busy = threading.Lock()
    
    def _hotkey_to_spec(self, combo: str) -> str:
        """Convert hotkey string to pynput format."""
        mapping = {
            "ctrl": "<ctrl>", "alt": "<alt>", 
            "shift": "<shift>", "win": "<cmd>", "cmd": "<cmd>"
        }
        parts = [mapping.get(part.strip().lower(), part.strip().lower()) 
                for part in combo.split("+")]
        return "+".join(parts)
    
    def on_ocr(self):
        """Handle OCR hotkey press."""
        if self._busy.locked():
            log.info("Already processing a capture")
            return
        
        def process_ocr():
            with self._busy:
                try:
                    if markdown := capture_and_transcribe(self.tokenizer, self.model, self.config):
                        copy_to_clipboard(markdown)
                        log.info("Copied Markdown to clipboard")
                except Exception as e:
                    log.exception("Transcription error: %s", e)
        
        threading.Thread(target=process_ocr, daemon=True).start()
    
    def on_quit(self):
        """Handle quit hotkey press."""
        log.info("Exiting on hotkey")
        if self.listener:
            self.listener.stop()
    
    def run(self):
        """Start the hotkey listener."""
        log.info("Ready. Press %s to OCR; %s to quit", 
                self.config["hotkey_ocr"], 
                self.config["hotkey_quit"])
        
        from pynput.keyboard import GlobalHotKeys
        
        hotkeys = {
            self._hotkey_to_spec(self.config["hotkey_ocr"]): self.on_ocr,
            self._hotkey_to_spec(self.config["hotkey_quit"]): self.on_quit,
        }
        
        with GlobalHotKeys(hotkeys) as listener:
            self.listener = listener
            listener.join()


def main():
    """Main entry point."""
    cfg = load_config("config.json")
    tokenizer, model = load_ocr_model(cfg)
    app = OCRHotkeyApp(tokenizer, model, cfg)
    app.run()


if __name__ == "__main__":
    main()