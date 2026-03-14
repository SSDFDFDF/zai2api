# z-ai2api_python

基于 FastAPI + Granian 的 GLM 代理服务，旨在将网页端 GLM 能力桥接到 OpenAI / Anthropic 兼容接口。
适合本地开发、自托管代理、Token 池管理和主流 AI 客户端（如 Cursor、Claude Code、Cherry Studio 等）接入。

## ✨ 特性

- **多协议兼容**：完美模拟 `OpenAI`、`Anthropic` 及 `Claude Code` 风格接口。
- **全系列模型支持**：支持 GLM-4.5、GLM-4.7、甚至是最新的 GLM-5 系列模型。
- **深度特性支持**：
    - **Thinking**：支持 GLM 推理增强版，实时输出思维链。
    - **Search**：集成联网搜索能力（基础搜索与高级搜索）。
    - **Vision**：支持图片上传与多模态交互。
    - **Tool Use**：采用 Toolify XML 方案，稳健支持工具调用。
- **智能 Token 池管理**：
    - **双池调度**：支持导入用户 Token 池与 Guest 匿名会话池。
    - **自动维护**：内置健康检查、失败熔断、自动恢复与去重功能。
    - **会话持久化**：支持 Chat ID 绑定，确保长对话上下文一致。
- **可视化管理后台**：
    - **仪表盘**：实时监控服务状态与请求量。
    - **Token 管理**：便捷导入（支持目录批量导入）与状态维护。
    - **配置管理**：支持在线修改环境变量并实时生效。
    - **实时日志**：后台直接查看格式化的请求与系统日志。
- **高性能部署**：采用 Granian 异步服务器，支持 Docker / Docker Compose 一键部署。

## 🚀 快速开始

### 环境要求

- Python `3.9` 到 `3.12`
- 推荐使用 [uv](https://github.com/astral-sh/uv)

### 本地启动

```bash
git clone https://github.com/ZyphrZero/z.ai2api_python.git
cd z.ai2api_python

uv sync
cp .env.example .env
# 根据需要修改 .env
uv run python main.py
```

首次启动会自动初始化数据库。默认地址：

- **API 根路径**：`http://127.0.0.1:8080`
- **Swagger 文档**：`http://127.0.0.1:8080/docs`
- **管理后台**：`http://127.0.0.1:8080/admin` (默认密码: `admin123`)

### Docker 部署

```bash
# 使用 Docker Compose 快速部署
docker compose -f deploy/docker-compose.yml up -d --build
```

更多部署说明见 [deploy/README_DOCKER.md](deploy/README_DOCKER.md)。

## ⚙️ 核心配置

建议在 `.env` 或管理后台中确认以下关键配置：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `AUTH_TOKEN` | `sk-your-api-key` | 客户端访问本服务使用的 Bearer Token |
| `ADMIN_PASSWORD` | `admin123` | 管理后台登录密码，**生产环境务必修改** |
| `ANONYMOUS_MODE` | `true` | 是否启用匿名模式（当 Token 池为空时回退 Guest 会话） |
| `UPSTREAM_USER_NAME` | `Guest` | 上游个性化变量：用户名称 |
| `TOOL_STRATEGY` | `xmlfc` | 工具调用策略：xmlfc / native / hybrid / glmnative / disabled |
| `MAX_UPLOAD_FILE_SIZE` | `10485760` | 最大文件上传限制 (10MB) |
| `DB_PATH` | `tokens.db` | SQLite 数据库路径 |

完整配置说明请参考 [.env.example](.env.example)。

## 🤖 兼容接口与模型

### 接口入口

- **OpenAI 兼容**：`/v1/chat/completions`
- **Anthropic 兼容**：`/v1/messages`
- **Claude Code 兼容**：`/anthropic/v1/messages`

### 支持模型映射

本项目内置了丰富的模型映射，您可以直接在客户端选择以下名称：

- **GLM-5 系列**：`GLM-5` (通用), `GLM-5-Thinking` (推理), `GLM-5-advanced-search` (高级搜索), `GLM-5-Agent` (智能体)
- **GLM-4.7 系列**：`GLM-4.7`, `GLM-4.7-Thinking`, `GLM-4.7-Search`, `GLM-4.7-advanced-search`
- **GLM-4.5 系列**：`GLM-4.5`, `GLM-4.5-Thinking`, `GLM-4.5-Search`, `GLM-4.5-Air`
- **视觉模型**：`GLM-4.6V` (支持多模态识图)

## 🖥️ 管理后台

管理后台提供了一站式的操作体验：

- `Dashboard`：掌握 Token 可用性及流量趋势。
- `Tokens`：支持手动添加、批量导入及活跃 Token 的实时负载状态。
- `Config`：无需重启，在线调整复杂的系统参数。
- `Logs`：查看带着色效果的即时请求日志，方便调试接口请求。

## 📜 常用维护命令

```bash
# 启动开发服务器
uv run python main.py

# 运行自动化测试
uv run pytest

# 数据库迁移 (Alembic)
uv run alembic upgrade head

# 静态检查 (Ruff)
uv run ruff check .
```

## ⚖️ 许可证及免责声明

- 本项目采用 **MIT 许可证**。
- **仅供学习和研究使用，严禁用于商业用途或违反上游服务条款的行为。**
- 本项目非官方产品，用户需自行承担使用风险。

---

<div align="center">
Made with ❤️ by the community
</div>
