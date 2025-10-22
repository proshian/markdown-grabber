import os
import sys
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
import tkinter as tk

from PIL import ImageGrab, Image
import torch
from transformers import AutoModel, AutoTokenizer
import pyperclip


log = logging.getLogger("markdowner")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[markdowner] %(levelname)s: %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


@dataclass
class Config:
    model: str = "deepseek-ai/DeepSeek-OCR"
    max_new_tokens: int = 2048
    temp: float = 0.1
    max_side: int = 1920
    load_4bit: bool = False
    system_prompt_path: str = "system_prompt.txt"
    system_prompt: str = ""
    hotkey_ocr: str = "ctrl+alt+m"
    hotkey_quit: str = "ctrl+alt+q"
    use_global_hotkeys: bool = True
    debug: bool = False
    use_system_snipping: bool = True
    # DeepSeek-OCR specific settings
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True
    test_compress: bool = True
    output_path: str = "ocr_output"
    use_flash_attention: bool = False


def load_config(path: str = "config.json") -> Config:
    cfg = Config()
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
        for k, v in data.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
    except Exception as e:
        log.warning("Using defaults; failed to read %s: %s", path, e)

    spath = cfg.system_prompt_path
    if not os.path.isabs(spath):
        spath = os.path.join(os.path.dirname(path), spath)
    try:
        cfg.system_prompt = open(spath, "r", encoding="utf-8").read().strip()
    except Exception as e:
        log.warning("No system prompt at %s (%s); using fallback.", spath, e)
        cfg.system_prompt = "<|grounding|>Convert the document to markdown."

    if os.getenv("MARKDOWNER_DEBUG"):
        cfg.debug = True
    log.setLevel(logging.DEBUG if cfg.debug else logging.INFO)
    log.info(f"Hotkeys -> OCR: {cfg.hotkey_ocr} | Quit: {cfg.hotkey_quit}")
    return cfg


def get_clipboard_image() -> Image.Image | None:
    data = ImageGrab.grabclipboard()
    if isinstance(data, Image.Image):
        return data
    if isinstance(data, list) and data:
        try:
            return Image.open(data[0])
        except Exception:
            return None
    return None


def wait_for_clipboard_image(timeout: float = 30.0, poll: float = 0.1) -> Image.Image | None:
    """Poll the clipboard for an image until timeout. Returns the image or None."""
    t0 = time.time()
    img = get_clipboard_image()
    while img is None and (time.time() - t0) < timeout:
        time.sleep(poll)
        img = get_clipboard_image()
    return img


def _invoke_windows_snipping() -> bool:
    # Prefer the modern Snipping Tool overlay via URI
    try:
        subprocess.Popen(["explorer.exe", "ms-screenclip:"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        pass
    # Fallback to legacy SnippingTool /clip if available
    try:
        subprocess.Popen(["SnippingTool", "/clip"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _invoke_macos_snipping() -> bool:
    try:
        # Interactive region capture to clipboard
        subprocess.Popen(["/usr/sbin/screencapture", "-i", "-c"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        try:
            subprocess.Popen(["screencapture", "-i", "-c"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False
    except Exception:
        return False


def _invoke_linux_snipping() -> bool:
    # Try GNOME first
    for cmd in (["gnome-screenshot", "-a", "-c"],  # area to clipboard
                ["gnome-screenshot", "-a" ],        # area to file (will not copy)
                ["xfce4-screenshooter", "-r", "-c"],
                ["spectacle", "-r", "-b", "-c"]):
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            continue
    return False


def invoke_system_snipping() -> bool:
    """Attempt to open the OS-native snipping overlay. Returns True if invoked."""
    plat = sys.platform
    if plat.startswith("win"):
        return _invoke_windows_snipping()
    if plat == "darwin":
        return _invoke_macos_snipping()
    # Assume Linux/Unix
    return _invoke_linux_snipping()


def select_region_fallback(timeout: float = 60.0) -> Image.Image | None:
    """Simple cross-platform region selector using Tkinter overlay as a fallback.
    Returns a PIL Image or None if cancelled.
    """
    bbox = {"x0": None, "y0": None, "x1": None, "y1": None}
    result: list[Image.Image] = []

    def on_press(event):
        bbox["x0"], bbox["y0"] = event.x, event.y
        canvas.delete("rect")

    def on_drag(event):
        x0, y0 = bbox["x0"], bbox["y0"]
        if x0 is None or y0 is None:
            return
        canvas.delete("rect")
        canvas.create_rectangle(x0, y0, event.x, event.y, outline="#00D1B2", width=2, tags="rect")

    def on_release(event):
        x0, y0 = bbox["x0"], bbox["y0"]
        if x0 is None or y0 is None:
            root.quit()
            return
        x1, y1 = event.x, event.y
        x0_, x1_ = sorted([x0, x1])
        y0_, y1_ = sorted([y0, y1])
        try:
            img = ImageGrab.grab(bbox=(x0_, y0_, x1_, y1_))
            result.append(img)
        except Exception as e:
            log.debug("ImageGrab fallback failed: %s", e)
        finally:
            root.quit()

    def on_escape(event):
        root.quit()

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    try:
        root.attributes("-alpha", 0.15)
    except Exception:
        pass
    root.configure(bg="black")
    root.attributes("-topmost", True)
    root.bind("<Escape>", on_escape)
    canvas = tk.Canvas(root, cursor="cross", highlightthickness=0, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)
    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)

    # Auto-cancel after timeout
    root.after(int(timeout * 1000), root.quit)
    try:
        root.mainloop()
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    return result[0] if result else None


def preprocess(img: Image.Image, max_side: int) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if max(img.size) > max_side > 0:
        s = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BICUBIC)
    return img


def copy_to_clipboard(text: str) -> None:
    pyperclip.copy(text)


def load_model_and_processor(cfg: Config):
    """Load DeepSeek-OCR model exactly like the working example"""
    log.info(f"Loading DeepSeek-OCR model: {cfg.model}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            cfg.model, 
            _attn_implementation="eager", # 'flash_attention_2', 
            trust_remote_code=True, 
            use_safetensors=True
        )
        
        # Set device and dtype exactly like working example
        if torch.cuda.is_available():
            model = model.eval().cuda().to(torch.bfloat16)
            log.info("Using CUDA with bfloat16 and flash_attention_2")
        else:
            model = model.eval().to(torch.float32)
            log.info("Using CPU with float32")
        
        log.info("DeepSeek-OCR model loaded successfully")
        return tokenizer, model
        
    except Exception as e:
        log.error(f"Failed to load DeepSeek-OCR model: {e}")
        raise


def save_temp_image(img: Image.Image) -> str:
    """Save image to temporary file for DeepSeek-OCR processing"""
    import tempfile
    temp_dir = "temp_ocr"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"capture_{int(time.time())}.png")
    img.save(temp_path, "PNG")
    return temp_path


def transcribe_image(img: Image.Image, tokenizer, model, cfg: Config) -> str:
    """Transcribe image using DeepSeek-OCR following the working example"""
    img = preprocess(img, cfg.max_side)
    
    # Save image to temporary file
    temp_image_path = save_temp_image(img)
    
    try:
        # Use the exact same parameters as your working example
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        
        log.debug("Starting OCR inference with working parameters...")
        
        # Create output directory
        os.makedirs(cfg.output_path, exist_ok=True)
        
        # Run inference with the exact same parameters that work
        res = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=temp_image_path,
            output_path=cfg.output_path,
            base_size=1024,
            image_size=640, 
            crop_mode=True,
            save_results=True,
            test_compress=True
        )
        
        # The infer method returns the result directly in your working example
        if res is not None:
            if isinstance(res, str):
                return res.strip()
            elif isinstance(res, (list, tuple)) and len(res) > 0:
                return str(res[0]).strip()
        
        # If res is None, check if there are output files
        log.debug("Checking for output files...")
        markdown_text = check_output_files(cfg.output_path, temp_image_path)
        if markdown_text:
            return markdown_text
            
        log.warning("No result returned from infer method")
        return "No OCR result obtained"
            
    except Exception as e:
        log.error(f"OCR inference failed: {e}")
        return f"Error during OCR: {str(e)}"
    finally:
        # Clean up temporary file
        try:
            os.remove(temp_image_path)
        except:
            pass

def check_output_files(output_path: str, image_path: str) -> str:
    """Check output directory for .mmd files specifically"""
    try:
        log.debug(f"Searching for MMD files in: {output_path}")
        
        # Search for all .mmd files recursively
        mmd_files = []
        for root, dirs, files in os.walk(output_path):
            for file in files:
                if file.endswith('.mmd'):
                    mmd_files.append(os.path.join(root, file))
        
        log.debug(f"Found MMD files: {mmd_files}")
        
        # Read the most recently modified .mmd file
        if mmd_files:
            # Sort by modification time, newest first
            mmd_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_mmd = mmd_files[0]
            log.debug(f"Reading latest MMD file: {latest_mmd}")
            
            with open(latest_mmd, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    log.debug(f"Found content in {latest_mmd}: {content[:500]}...")
                    return content
                else:
                    log.debug("MMD file is empty")
        
        # Fallback: look for any text files
        log.debug("No MMD files found, searching for any text files...")
        for root, dirs, files in os.walk(output_path):
            for file in files:
                if file.endswith(('.txt', '.md', '.markdown')):
                    file_path = os.path.join(root, file)
                    log.debug(f"Reading text file: {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            log.debug(f"Found content in {file_path}: {content[:500]}...")
                            return content
                            
    except Exception as e:
        log.error(f"Error checking output files: {e}")
    
    return ""



def cleanup_old_outputs(output_path: str, keep_recent: int = 3):
    """Clean up old output directories, keeping only the most recent ones"""
    try:
        if not os.path.exists(output_path):
            return
            
        # Find all result_* directories
        result_dirs = []
        for item in os.listdir(output_path):
            item_path = os.path.join(output_path, item)
            if os.path.isdir(item_path) and item.startswith('result_'):
                result_dirs.append((item_path, os.path.getmtime(item_path)))
        
        # Sort by modification time, newest first
        result_dirs.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old directories (keep only the most recent ones)
        for dir_path, _ in result_dirs[keep_recent:]:
            import shutil
            shutil.rmtree(dir_path)
            log.debug(f"Cleaned up old output: {dir_path}")
            
    except Exception as e:
        log.debug(f"Could not clean up old outputs: {e}")


def capture_then_transcribe(tokenizer: AutoTokenizer, model: AutoModel, cfg: Config) -> str | None:
    """Invoke a snipping tool, wait for the clipboard image (or fallback selector), transcribe, return Markdown."""
    if not cfg.use_system_snipping:
        log.info("Using built-in selector (system snipping disabled).")
        img = select_region_fallback(timeout=60.0)
        if img is None:
            log.warning("Screenshot cancelled or unavailable.")
            return None
        return transcribe_image(img, tokenizer, model, cfg)
    
    invoked = invoke_system_snipping()
    img: Image.Image | None = None
    if invoked:
        log.debug("Snipping tool invoked, waiting for clipboard image...")
        img = wait_for_clipboard_image(timeout=60.0, poll=0.15)
    if img is None:
        log.info("Using built-in selector fallback.")
        img = select_region_fallback(timeout=60.0)
    if img is None:
        log.warning("Screenshot cancelled or unavailable.")
        return None
    return transcribe_image(img, tokenizer, model, cfg)


class HotkeyApp:
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModel, config: Config):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.listener = None
        self._busy = threading.Lock()

    def on_ocr(self) -> None:
        # Run asynchronously to avoid blocking global hotkeys
        if self._busy.locked():
            log.info("Already processing a capture. Please wait...")
            return

        def worker():
            with self._busy:
                try:
                    markdown = capture_then_transcribe(self.tokenizer, self.model, self.config)
                    if markdown:
                        copy_to_clipboard(markdown)
                        log.info("Copied Markdown to clipboard.")
                except Exception as exc:
                    log.exception("Transcription error: %s", exc)

        threading.Thread(target=worker, daemon=True).start()

    def on_quit(self) -> None:
        log.info("Exiting on hotkey.")
        if self.listener:
            self.listener.stop()

    def to_spec(self, combo: str) -> str:
        mapping = {"ctrl": "<ctrl>", "alt": "<alt>", "shift": "<shift>", "win": "<cmd>", "cmd": "<cmd>"}
        return "+".join(mapping.get(part, part) for part in (x.strip().lower() for x in combo.split("+")) if part)

    def run(self) -> None:
        log.info(f"Ready. Press {self.config.hotkey_ocr} to OCR; {self.config.hotkey_quit} to quit.")
        from pynput.keyboard import GlobalHotKeys
        
        hotkeys = {
            self.to_spec(self.config.hotkey_ocr): self.on_ocr,
            self.to_spec(self.config.hotkey_quit): self.on_quit,
        }
        with GlobalHotKeys(hotkeys) as listener:
            self.listener = listener
            listener.join()


def main():
    cfg = load_config("config.json")
    tokenizer, model = load_model_and_processor(cfg)
    app = HotkeyApp(tokenizer, model, cfg)
    app.run()


if __name__ == "__main__":
    main()