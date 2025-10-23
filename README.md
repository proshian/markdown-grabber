# Markdowner - Screenshot to Markdown with DeepSeek OCR

The snapping tool that gives you text instead of images. Powered by DeepSeek OCR for exceptional accuracy on text, code, and math.

![Markdowner demo](https://github.com/user-attachments/assets/f6bef1fe-d50a-4604-a76a-1d1b0069dedb)

## 🚀 How It Works

**Simple workflow:**
1. **Capture** (`Ctrl + Alt + C`) - Select any screen area
2. DeepSeek-OCR automatically extracts text as markdown  
3. **Paste** (`Ctrl + V`) - Use the formatted text anywhere

**Ideal for:**
- 📝 Extracting text from documentation and articles
- 🧮 Converting math formulas to LaTeX with precision
- 📊 Turning tables and data into clean markdown
- 💻 Copying code snippets from images or videos
- 📋 Digitizing printed text or handwritten notes

## ⚡ Get Started

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

## Seamless Workflow

This app is built to eliminate the constant context switching when extracting text from images. The old workflow was always disruptive:

**Before:**
- Switch away from your current app
- Open a browser or external tool
- Upload screenshots and wait for processing
- Manually copy results back to your workspace
- Lose focus and break your workflow

**Now:**
- Press `Ctrl + Alt + C` → select area
- Press `Ctrl + V` → text is ready to use

The key advantage: you never leave your workspace. No alt-tabbing, no browser tabs, no context switching. The tool works silently in the background, letting you maintain focus while seamlessly converting images to text right where you need it.
