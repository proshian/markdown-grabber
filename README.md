# markdowner

Convert screenshots to Markdown using a local AI model with just a few hotkeys.


## Setup
Set up a virtual environment and install dependencies:

```sh
pip install uv  
uv sync
```

There are other ways to install uv, see [documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Quick Start

Run with `uv run markdowner.py`

You can also launch the script via a shortcut. For example, on Windows:

1. Create a shortcut to `markdowner_looped.bat`.
2. In the shortcut properties set Shortcut key: `Ctrl + Alt + L`.
3. Launch with `Ctrl + Alt + L`. Stop with `Ctrl + Alt + Q`.

## Hotkey cycle
1. `Shift + Win + S` → capture area  
2. `Shift + Alt + M` → convert to Markdown and push to clipboard
3. `Ctrl + V` → paste  (Though we recommend `Win + V --> Enter` to easily detect the moment the text appears in the clipboard. Note that `Win + V` works only on Windows 10/11 and requires enabling clipboard history in Windows settings.)

Stop the loop with `Ctrl + Alt + Q`.


# TODO
- [ ] Check support for Linux and MacOS
- [ ] If possible make it so that if the loop is not running you press Ctrl + Alt + M to start the loop and start the task. So that there's no Ctrl + Alt + L step.
