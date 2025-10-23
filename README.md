Markdowner - Screenshot to Markdown with DeepSeek OCR

The snapping tool that gives you text instead of images. Powered by DeepSeek OCR for exceptional accuracy on text, code, and math.

![Markdowner demo](https://github.com/user-attachments/assets/f6bef1fe-d50a-4604-a76a-1d1b0069dedb)



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


## Seamless Workflow Integration

I wanted to eliminate the constant context switching when working with text from images. The old process was always disruptive:

**Before:**
- Switch away from your current app
- Open a browser or another tool  
- Upload the screenshot
- Wait for processing
- Copy the results back
- Return to your original work

**Now:**
- Press `Ctrl + Alt + C` → select area
- Press `Ctrl + V` → use the text

The key is that you never leave your workspace. No switching windows, no waiting for external tools - just capture and continue working. The tool stays out of your way while handling the conversion automatically.
