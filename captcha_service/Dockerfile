# ---------------------------------------------------------------------------
# Z.ai Captcha Token Service — Hugging Face Spaces Dockerfile
#
# 部署到 Hugging Face Spaces 时选择 "Docker" SDK 并指向此文件。
# Space 会自动暴露 7860 端口，无需额外配置。
# ---------------------------------------------------------------------------

FROM python:3.12-slim

# Playwright Chromium 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libdbus-1-3 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装 Chromium 浏览器（不带系统依赖，已手动安装）
RUN python -m playwright install chromium --with-deps || true
RUN python -m playwright install chromium

COPY server.py .

# Hugging Face Spaces 固定使用 7860 端口
ENV CAPTCHA_HOST=0.0.0.0
ENV CAPTCHA_PORT=7860
ENV CAPTCHA_HEADLESS=true

CMD ["python", "server.py"]
