#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Z.ai Captcha Token Service — Startup Script
#
# Automatically detects and installs Playwright browser if needed,
# then starts the server.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Configuration (override via environment)
# ---------------------------------------------------------------------------
CAPTCHA_PORT="${CAPTCHA_PORT:-9000}"
CAPTCHA_HOST="${CAPTCHA_HOST:-0.0.0.0}"
CAPTCHA_HEADLESS="${CAPTCHA_HEADLESS:-true}"
CAPTCHA_TOKEN_TIMEOUT="${CAPTCHA_TOKEN_TIMEOUT:-15.0}"
CAPTCHA_PAGE_IDLE_TTL="${CAPTCHA_PAGE_IDLE_TTL:-300}"

export CAPTCHA_HOST CAPTCHA_PORT CAPTCHA_HEADLESS
export CAPTCHA_TOKEN_TIMEOUT CAPTCHA_PAGE_IDLE_TTL

# ---------------------------------------------------------------------------
# Python virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[captcha] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
if ! python -c "import fastapi" 2>/dev/null; then
    echo "[captcha] Installing Python dependencies..."
    pip install -r "$SCRIPT_DIR/requirements.txt" -q
fi

# ---------------------------------------------------------------------------
# Playwright browser detection & installation
# ---------------------------------------------------------------------------
PLAYWRIGHT_BROWSERS_DIR="${PLAYWRIGHT_BROWSERS_DIR:-$HOME/.cache/ms-playwright}"

check_chromium_installed() {
    if [ -d "$PLAYWRIGHT_BROWSERS_DIR" ]; then
        # Check for any chromium installation
        if ls "$PLAYWRIGHT_BROWSERS_DIR"/chromium-*/chrome-linux/chrome 2>/dev/null | head -1 >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

if check_chromium_installed; then
    CHROMIUM_PATH=$(ls -d "$PLAYWRIGHT_BROWSERS_DIR"/chromium-*/chrome-linux/chrome 2>/dev/null | head -1)
    echo "[captcha] Chromium found: $CHROMIUM_PATH"
else
    echo "[captcha] Chromium not found. Installing Playwright browser..."
    python -m playwright install chromium
    if check_chromium_installed; then
        echo "[captcha] Chromium installed successfully."
    else
        echo "[captcha] ERROR: Failed to install Chromium."
        echo "[captcha] Try manually: python -m playwright install chromium --with-deps"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# System dependencies check (Linux only)
# ---------------------------------------------------------------------------
if [ "$(uname -s)" = "Linux" ]; then
    MISSING_DEPS=""
    for lib in libnss3.so libnspr4.so libatk-1.0.so libatk-bridge-2.0.so \
               libcups.so libdrm.so libdbus-1.so libxkbcommon.so \
               libxcomposite.so libxdamage.so libxfixes.so libxrandr.so \
               libgbm.so libpango-1.0.so libcairo.so libasound.so; do
        if ! ldconfig -p 2>/dev/null | grep -q "$lib"; then
            # Also try checking via dpkg/pkg-config
            if ! dpkg -l 2>/dev/null | grep -q "$lib"; then
                MISSING_DEPS="$MISSING_DEPS  $lib"
            fi
        fi
    done
    if [ -n "$MISSING_DEPS" ]; then
        echo "[captcha] WARNING: Some system libraries may be missing:$MISSING_DEPS"
        echo "[captcha] If Chromium fails to start, run:"
        echo "[captcha]   playwright install-deps chromium"
    fi
fi

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------
echo "[captcha] Starting Captcha Token Service on $CAPTCHA_HOST:$CAPTCHA_PORT..."
echo "[captcha] Headless: $CAPTCHA_HEADLESS"
echo "[captcha] Token timeout: ${CAPTCHA_TOKEN_TIMEOUT}s"
echo "[captcha] Page idle TTL: ${CAPTCHA_PAGE_IDLE_TTL}s"

exec python "$SCRIPT_DIR/server.py"
