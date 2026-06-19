FROM node:20-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    chromium \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

COPY captcha-provider/package*.json ./
RUN npm ci --omit=dev

COPY captcha-provider/server.js ./

ENV HOST=0.0.0.0
ENV PORT=9876
ENV BROWSER_BACKEND=playwright
ENV CHROMIUM_PATH=/usr/bin/chromium

EXPOSE 9876

CMD ["node", "server.js"]
