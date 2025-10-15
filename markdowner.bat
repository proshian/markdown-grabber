@echo off
cd /d %~dp0

rem Single-instance guard via Windows named mutex (no deps)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$m = [System.Threading.Mutex]::new($false,'Global\markdowner_singleton');" ^
  "if(-not $m.WaitOne(0)) { Write-Host '[markdowner] Another instance is already running.'; exit 1 }" ^
  "try { uv run 'markdowner.py' } finally { $m.ReleaseMutex(); $m.Dispose() }"

exit /b %ERRORLEVEL%