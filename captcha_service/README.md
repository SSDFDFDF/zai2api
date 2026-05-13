# Z.ai Captcha Token Service

独立的 Captcha Token 生成服务，通过 Playwright 浏览器自动化与阿里云验证码 SDK 交互，为主 API Proxy 提供 `captcha_verify_param`。

## 架构

```
主 API Proxy (main.py) ──HTTP──► Captcha Service (server.py)
                                     │
                                     ├── Playwright Browser
                                     ├── chat.z.ai (JWT 注入)
                                     └── Aliyun Captcha SDK
```

- **独立部署**：可与主服务部署在不同服务器，避免内存叠加
- **内存占用**：~300-500MB（Playwright + Chromium）
- **主服务轻量**：只需 `httpx` 调用，无需 Playwright 依赖

## 快速启动

```bash
cd captcha_service
bash start.sh
```

`start.sh` 会自动：
1. 创建 Python 虚拟环境
2. 安装依赖（`fastapi`, `uvicorn`, `playwright`）
3. 检测/安装 Chromium 浏览器
4. 检查 Linux 系统依赖（libnss3 等）
5. 启动服务

## 手动启动

```bash
cd captcha_service
source .venv/bin/activate
python server.py
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CAPTCHA_HOST` | `0.0.0.0` | 监听地址 |
| `CAPTCHA_PORT` | `9000` | 监听端口 |
| `CAPTCHA_HEADLESS` | `true` | 是否无头模式运行浏览器 |
| `CAPTCHA_TOKEN_TIMEOUT` | `15.0` | 获取 token 超时时间（秒） |
| `CAPTCHA_PAGE_IDLE_TTL` | `300` | 闲置页面自动关闭 TTL（秒） |

## API

### POST /token

生成 captcha token。

**请求**：
```json
{"jwt_token": "eyJ..."}
```

**响应**：
```json
{
  "captcha_verify_param": "eyJjZXJ...",
  "certify_id": "Z8Rg5UyvpO"
}
```

### GET /health

健康检查。

**响应**：
```json
{
  "status": "ok",
  "browser_connected": true,
  "active_pages": 3
}
```

## 工作原理

1. 从 JWT 解码 `user_id`，按用户维护独立浏览器页面
2. 每个页面注入 JWT 到 `localStorage`，模拟已登录的 chat.z.ai 会话
3. 获取 token 流程：填充 textarea → 按 Enter → Playwright route 拦截 → 捕获 `captcha_verify_param` → abort 请求（防止 token 被消耗）
4. 同一用户的并发请求由 `asyncio.Lock` 串行化
5. 闲置 5 分钟的页面自动清理

## Docker 部署

```dockerfile
FROM python:3.12-slim
RUN pip install fastapi uvicorn playwright pydantic
RUN playwright install chromium --with-deps
COPY server.py .
CMD ["python", "server.py"]
```

```bash
docker build -t captcha-service .
docker run -d -p 9000:9000 captcha-service
```
