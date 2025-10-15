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
from transformers import (
    AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq,
    AutoModelForImageTextToText, AutoConfig, BitsAndBytesConfig,
    PreTrainedModel, ProcessorMixin,
)
from pynput.keyboard import GlobalHotKeys
import pyperclip


log = logging.getLogger("markdowner")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[markdowner] %(levelname)s: %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
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
        cfg.system_prompt = "You are an OCR-to-Markdown transcriber. Output only Markdown."

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


def load_model_and_processor(cfg: Config) -> tuple[ProcessorMixin, PreTrainedModel]:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = {"torch_dtype": dtype, "device_map": "auto", "low_cpu_mem_usage": True, "trust_remote_code": True}
    if cfg.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    processor = AutoProcessor.from_pretrained(cfg.model, trust_remote_code=True, use_fast=True)
    mcfg = AutoConfig.from_pretrained(cfg.model, trust_remote_code=True)
    is_vl = (getattr(mcfg, "model_type", "") or "").lower().find("vl") >= 0 or hasattr(mcfg, "vision_config")
    if is_vl:
        try:
            model = AutoModelForImageTextToText.from_pretrained(cfg.model, **kwargs)
        except Exception:
            model = AutoModelForVision2Seq.from_pretrained(cfg.model, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model, **kwargs)
    model.eval()
    return processor, model


def build_messages(img: Image.Image, cfg: Config):
    return [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": [{"type": "image", "image": img},
                                     {"type": "text", "text": "Transcribe this image to Markdown. Output only Markdown."}]},
    ]


def transcribe_image(img: Image.Image, processor: ProcessorMixin, model: PreTrainedModel, cfg: Config) -> str:
    img = preprocess(img, cfg.max_side)
    msgs = build_messages(img, cfg)
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    gen_kwargs = {"max_new_tokens": cfg.max_new_tokens, "temperature": cfg.temp, "do_sample": cfg.temp > 0}
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    new_ids = out[:, inputs["input_ids"].shape[1]:]
    md = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return md


def capture_then_transcribe(processor: ProcessorMixin, model: PreTrainedModel, cfg: Config) -> str | None:
    """Invoke a snipping tool, wait for the clipboard image (or fallback selector), transcribe, return Markdown."""
    if not cfg.use_system_snipping:
        log.info("Using built-in selector (system snipping disabled).")
        img = select_region_fallback(timeout=60.0)
        if img is None:
            log.warning("Screenshot cancelled or unavailable.")
            return None
        return transcribe_image(img, processor, model, cfg)
    
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
    return transcribe_image(img, processor, model, cfg)


class HotkeyApp:
    def __init__(self, processor: ProcessorMixin, model: PreTrainedModel, config: Config):
        self.processor = processor
        self.model = model
        self.config = config
        self.listener: GlobalHotKeys | None = None
        self._busy = threading.Lock()

    def on_ocr(self) -> None:
        # Run asynchronously to avoid blocking global hotkeys
        if self._busy.locked():
            log.info("Already processing a capture. Please wait...")
            return

        def worker():
            with self._busy:
                try:
                    markdown = capture_then_transcribe(self.processor, self.model, self.config)
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
        hotkeys = {
            self.to_spec(self.config.hotkey_ocr): self.on_ocr,
            self.to_spec(self.config.hotkey_quit): self.on_quit,
        }
        with GlobalHotKeys(hotkeys) as listener:
            self.listener = listener
            listener.join()


def main():
    cfg = load_config("config.json")
    processor, model = load_model_and_processor(cfg)
    app = HotkeyApp(processor, model, cfg)
    app.run()


if __name__ == "__main__":
    main()
    