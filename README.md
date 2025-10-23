# Markdowner - Screenshot to Markdown with DeepSeek OCR

OCR app powered by DeepSeek-OCR that allows you to convert anything to markdown with just a single hotkey

![Demo GIF](demo.gif)

## 🚀 What Does It Do?

**Simple workflow:**
1. Press `Ctrl + Alt + C` → Select screen area
2. DeepSeek-OCR converts image to markdown automatically  
3. Press `Ctrl + V` → Paste the text anywhere

**Perfect for:**
- 📝 Extracting text from documentation
- 🧮 Converting math formulas to LaTeX
- 📊 Turning tables into markdown
- 💻 Copying code snippets from images

## ⚡ Quick Start

```bash
pip install uv
uv sync
uv run markdowner.py
```

> There are other ways to install uv, see [documentation](https://docs.astral.sh/uv/getting-started/installation/).

**Hotkeys:**
- `Ctrl + Alt + C` - Capture screen area → markdown
- `Ctrl + V` - Paste the converted text
- `Ctrl + Alt + Q` - Quit the app

>[!Tip]
> The hotkeys can be adjusted in `config.json`

>[!Tip]
> You can also create a shortcut to launch the app itself.
>
> For example, to do so on Windows:
>
> 1. Create a shortcut to `markdowner_looped.bat`.
> 2. In the shortcut properties set Shortcut key: `Ctrl + Alt + L`.
> 3. Launch with `Ctrl + Alt + L`. Stop with `Ctrl + Alt + Q`.

## Why I build this?

Without this app I used to go to an AI assistant to parse the long screenshots and for short formulas I used to type them myself. It was time-consuming and frustrating. Now it's literally "copy and paste".
