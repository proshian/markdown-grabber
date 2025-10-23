import json
import logging
import threading
import tempfile
from pathlib import Path
from typing import Optional
import re

import tkinter as tk
from PIL import ImageGrab, Image, ImageTk
import torch
from transformers import AutoModel, AutoTokenizer
from pynput.keyboard import GlobalHotKeys


logging.basicConfig(
    level=logging.INFO,
    format="[markdowner] %(levelname)s: %(message)s"
)
log = logging.getLogger("markdowner")


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_clipboard_image() -> Optional[Image.Image]:
    return ImageGrab.grabclipboard()


class RegionSelector:
    """
    Cross-platform region selector using Tkinter.
    Displays a dimmed overlay and highlights the selected region.
    """

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.result: Optional[Image.Image] = None

        # These MUST be instance attributes to prevent garbage collection
        self.tk_img_full: Optional[ImageTk.PhotoImage] = None  # Full screenshot
        self.tk_img_overlay: Optional[ImageTk.PhotoImage] = None  # Dimming overlay
        self.tk_img_cropped: Optional[ImageTk.PhotoImage] = None  # "Clear" selection
        # ---

    def _on_press(self, event, bbox: dict, canvas: tk.Canvas):
        """Handles the start of a mouse click/drag."""
        bbox["x0"], bbox["y0"] = event.x, event.y
        # Delete any previous selection drawings
        canvas.delete("selection")

    def _on_drag(self, event, bbox: dict, canvas: tk.Canvas, full_ss: Image.Image):
        """Handles the mouse drag, drawing the "clear" rectangle."""
        if bbox["x0"] is None or bbox["y0"] is None:
            return

        canvas.delete("selection")

        x0, y0 = bbox["x0"], bbox["y0"]
        x1, y1 = event.x, event.y

        # Ensure coordinates are ordered
        rx0, rx1 = sorted([x0, x1])
        ry0, ry1 = sorted([y0, y1])
        
        # Prevent zero-width/height crop
        if rx0 == rx1: rx1 += 1
        if ry0 == ry1: ry1 += 1

        try:
            # Crop the *original* screenshot
            cropped_img = full_ss.crop((rx0, ry0, rx1, ry1))

            # Convert to PhotoImage and store it
            # This reference MUST be kept to prevent garbage collection
            self.tk_img_cropped = ImageTk.PhotoImage(cropped_img)

            # Draw the "clear" (original) cropped image on top
            canvas.create_image(
                rx0, ry0, image=self.tk_img_cropped,
                anchor=tk.NW, tags="selection"
            )

            # Draw the selection border
            canvas.create_rectangle(
                rx0, ry0, rx1, ry1,
                outline="#00D1B2", width=2, tags="selection"
            )
        except Exception as e:
            log.debug("Failed to create/draw selection: %s", e)

    def _on_release(self, event, bbox: dict, root: tk.Tk, full_ss: Image.Image):
        """Handles mouse release, finalizing the selection."""
        if bbox["x0"] is None or bbox["y0"] is None:
            root.quit()
            return

        x0, y0 = bbox["x0"], bbox["y0"]
        x1, y1 = event.x, event.y
        
        rx0, rx1 = sorted([x0, x1])
        ry0, ry1 = sorted([y0, y1])

        try:
            # Crop the original screenshot to get the final result
            self.result = full_ss.crop((rx0, ry0, rx1, ry1))
        except Exception as e:
            log.debug("Image crop failed: %s", e)
        finally:
            root.quit()

    def select(self) -> Optional[Image.Image]:
        """Show region selector and return captured image."""
        
        # 1. Take the full screenshot *before* creating the window
        try:
            full_ss = ImageGrab.grab()
        except Exception as e:
            log.debug("Full ImageGrab failed: %s", e)
            return None

        root = tk.Tk()
        root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)
        # We NO LONGER make the window transparent
        # root.attributes("-alpha", 0.15) 

        # 2. Convert full screenshot to a Tkinter-compatible image
        # Store as instance variable to prevent garbage collection
        self.tk_img_full = ImageTk.PhotoImage(full_ss)

        # 3. Create the semi-transparent overlay image
        # Using 30% opacity black (RGBA). Your 0.15 alpha was very light.
        # Adjust 77 (30% opacity) to 38 (15% opacity) if you prefer.
        overlay_color = (0, 0, 0, 77)  # Black, 30% opaque
        overlay_img = Image.new("RGBA", full_ss.size, overlay_color)
        self.tk_img_overlay = ImageTk.PhotoImage(overlay_img)

        # 4. Create canvas and draw the images
        canvas = tk.Canvas(root, cursor="cross", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        # Draw the full screenshot (base layer)
        canvas.create_image(0, 0, image=self.tk_img_full, anchor=tk.NW)
        
        # Draw the dimming overlay on top
        canvas.create_image(0, 0, image=self.tk_img_overlay, anchor=tk.NW, tags="overlay")

        bbox = {"x0": None, "y0": None}

        # 5. Bind events, passing the full_ss to drag/release handlers
        canvas.bind(
            "<ButtonPress-1>", 
            lambda e: self._on_press(e, bbox, canvas)
        )
        canvas.bind(
            "<B1-Motion>", 
            lambda e: self._on_drag(e, bbox, canvas, full_ss)
        )
        canvas.bind(
            "<ButtonRelease-1>", 
            lambda e: self._on_release(e, bbox, root, full_ss)
        )
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
