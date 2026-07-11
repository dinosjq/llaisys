---
name: wsl-execution
description: How to execute WSL-side Python from Windows-side Claude Code session
metadata:
  type: project
---

To run Python scripts inside WSL from a Windows-side Claude Code session (Git Bash shell), prefix with `MSYS_NO_PATHCONV=1 wsl -e` and use full Linux paths:

```bash
export MSYS_NO_PATHCONV=1
wsl -e /home/songjq/llaisys-main/venv/bin/python /home/songjq/llaisys-main/path/to/script.py --args 2>/dev/null
```

**Why:** Claude Code on Windows uses Git Bash, which auto-converts paths like `/home/...` to Windows paths (`C:/Program Files/Git/home/...`). `MSYS_NO_PATHCONV=1` blocks this conversion. The WSL banner messages are UTF-16 encoded and appear as garbled text on stdout — redirect stderr with `2>/dev/null` to suppress them.

**How to apply:** Any time a Python script needs the LLAISYS shared library (`.so`), use this pattern. The Windows Python interpreter can only load `.dll` files; WSL Python loads `.so` files. The LLAISYS C++ backend must be built with `xmake` inside WSL first.

Note: `xmake` cannot run from the Windows side (lock-file error on UNC paths), so compilation must be done in a native WSL terminal.
