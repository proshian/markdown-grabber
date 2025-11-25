# Markdowner - Screenshot to Markdown with HunyuanOCR

HunyuanOCR-powered screenshot OCR tool that extracts text to your clipboard

![markdowner_demo](https://github.com/user-attachments/assets/5c8e2eec-4b92-4fbc-936f-1296cb7daf45)


## Seamless Workflow

I was tired of being pulled away from my workspace to convert screenshots. This tool is the solution: a simple hotkey (Ctrl+Alt+C) captures any screen area and copies the extracted markdown text directly to my clipboard. Now I can paste instantly and stay focused.

## 🚀 How It Works

**Simple workflow:**
- Press `Ctrl + Alt + C` → select a screen area (HunyuanOCR extracts the text in the background and it's copied to the clipboard)
- Press `Ctrl + V` → paste the text

**Ideal for:**
- 📝 Extracting text from documentation and articles
- 🧮 Converting math formulas to LaTeX with precision
- 📊 Turning tables and data into clean markdown
- 💻 Copying code snippets from images or videos
- 📋 Digitizing printed text or handwritten notes

## Get Started

```bash
pip install uv
uv sync
uv run markdowner.py
```

> There are other ways to install uv, see [documentation](https://docs.astral.sh/uv/getting-started/installation/).

**Essential Hotkeys:**
- `Ctrl + Alt + C` - Capture screen area and convert to markdown
- `Ctrl + V` - Paste the converted text
- `Ctrl + Alt + Q` - Quit the application

>[!Tip]
> Customize hotkeys in `config.json` to match your preferences

>[!Tip]
> **Quick Launch Setup (Windows):**
> 1. Create a shortcut to `markdowner_looped.bat`
> 2. Set shortcut key to `Ctrl + Alt + L` in properties
> 3. Now you can launch instantly with `Ctrl + Alt + L`


## Journey:
1. **Qwen VLM** - Initial attempt, but accuracy wasn't satisfactory
2. **DeepSeek-OCR** - Excellent accuracy, but too slow and resource-intensive
3. **HunyuanOCR** - The sweet spot: fast, accurate, and efficient (requires just 6Gb VRAM)

Now the app works flawlessly, and I don't need to leave my workspace
