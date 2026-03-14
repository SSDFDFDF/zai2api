# Docker 部署文档

## 快速部署

### 方式一: 使用预构建镜像 (推荐)

从 Docker Hub 拉取镜像:

```bash
# 拉取最新镜像
docker pull zyphrzero/z-ai2api-python:latest

# 创建数据目录
mkdir -p data logs

# 快速启动
docker run -d \
  --name z-ai-api-server \
  -p 8080:8080 \
  -e ADMIN_PASSWORD=admin123 \
  -e AUTH_TOKEN=sk-your-api-key \
  -e ANONYMOUS_MODE=true \
  -e DB_PATH=/app/data/tokens.db \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --restart unless-stopped \
  zyphrzero/z-ai2api-python:latest
```

**优势**:
- ✅ 无需本地构建,节省时间
- ✅ GitHub Actions 自动化构建,保证质量
- ✅ 多架构支持 (amd64/arm64)
- ✅ 镜像已优化,体积更小

### 方式二: 使用本地构建

适用于需要自定义修改代码的场景:

```bash
# 进入部署目录
cd deploy

# 启动服务 (会自动构建镜像)
docker compose up -d

# 查看日志
docker compose logs -f api-server
```

服务将在 `http://localhost:8080` 启动。

## 架构说明

### 持久化存储

容器使用卷映射实现数据持久化:

```yaml
volumes:
  - ./data:/app/data      # 数据库存储 (tokens.db)
  - ./logs:/app/logs      # 应用日志
```

**目录结构**:
```
deploy/
├── data/
│   └── tokens.db          # SQLite 数据库 (自动创建)
├── logs/                  # 应用日志 (自动创建)
├── docker-compose.yml
├── Dockerfile
└── README_DOCKER.md
```

### 环境变量

核心配置参数 (在 `docker-compose.yml` 中设置):

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DB_PATH` | `/app/data/tokens.db` | 数据库文件路径 |
| `ADMIN_PASSWORD` | `admin123` | 管理后台密码 |
| `AUTH_TOKEN` | `sk-your-api-key` | API 认证密钥 |
| `SKIP_AUTH_TOKEN` | `false` | 跳过 API 验证 |
| `ANONYMOUS_MODE` | `true` | 匿名访问模式 |
| `DEBUG_LOGGING` | `true` | 调试日志开关 |
| `TOOL_STRATEGY` | `xmlfc` | 工具调用策略：xmlfc / native / hybrid / glmnative / disabled |

**生产环境建议**:
- 修改 `ADMIN_PASSWORD` 和 `AUTH_TOKEN`
- 设置 `DEBUG_LOGGING=false`
- 设置 `ANONYMOUS_MODE=false`

## 运维操作

### 服务管理

```bash
# 启动服务
docker compose up -d

# 停止服务
docker compose down

# 重启服务
docker compose restart

# 查看状态
docker compose ps

# 实时日志
docker compose logs -f
```

### 更新应用

**使用预构建镜像**:

```bash
# 停止当前容器
docker compose down

# 拉取最新镜像
docker pull zyphrzero/z-ai2api-python:latest

# 启动新版本 (数据会自动保留)
docker compose up -d

# 清理旧镜像
docker image prune -f
```

**使用本地构建**:

```bash
# 拉取最新代码
git pull

# 重新构建并启动 (数据会保留)
docker compose up -d --build

# 清理旧镜像
docker image prune -f
```

### 数据备份与恢复

**备份**:
```bash
# 备份数据库
cp ./data/tokens.db ./data/tokens.db.backup.$(date +%Y%m%d_%H%M%S)

# 完整备份
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz ./data ./logs
```

**恢复**:
```bash
# 停止服务
docker compose down

# 恢复数据库
cp ./data/tokens.db.backup.20250116_120000 ./data/tokens.db

# 启动服务
docker compose up -d
```

### 数据库迁移

如需从其他位置迁移现有数据库:

```bash
# 使用迁移脚本
./migrate_db.sh /path/to/existing/tokens.db

# 或手动复制
cp /opt/1panel/docker/compose/k2think/tokens.db ./data/
chmod 644 ./data/tokens.db

# 启动服务
docker compose up -d
```

## 故障排查

### 数据库初始化失败

**错误**: `unable to open database file`

**原因**: 目录权限或卷映射问题

**解决**:
```bash
# 停止容器
docker compose down

# 确保目录存在
mkdir -p ./data ./logs

# 设置权限
chmod 755 ./data ./logs

# 重新构建并启动
docker compose up -d --build
```

### 容器无法启动

**检查步骤**:
```bash
# 查看详细日志
docker compose logs api-server

# 检查容器状态
docker compose ps

# 验证配置文件
docker compose config
```

### 端口冲突

如端口 8080 被占用,修改 `docker-compose.yml`:
```yaml
ports:
  - "8081:8080"  # 映射到宿主机 8081 端口
```

### 健康检查失败

```bash
# 检查健康状态
docker compose ps

# 手动测试接口
curl http://localhost:8080/v1/models

# 进入容器排查
docker exec -it z-ai-api-server bash
```

## API 访问

| 端点 | 地址 | 说明 |
|------|------|------|
| API 根路径 | `http://localhost:8080` | OpenAI 兼容 API |
| 模型列表 | `http://localhost:8080/v1/models` | 获取可用模型 |
| 管理后台 | `http://localhost:8080/admin` | Web 管理界面 |
| API 文档 | `http://localhost:8080/docs` | OpenAPI/Swagger 文档 |
| 健康检查 | `http://localhost:8080/v1/models` | 服务健康状态 |

## 高级配置

### 自定义数据库路径

修改 `docker-compose.yml` 使用外部路径:

```yaml
volumes:
  - /opt/mydata:/app/data  # 使用绝对路径

environment:
  - DB_PATH=/app/data/tokens.db
```

### 使用 .env 文件

创建 `.env` 文件 (基于 `.env.example`):

```bash
cp .env.example .env
# 编辑配置
vim .env
```

修改 `docker-compose.yml`:
```yaml
services:
  api-server:
    env_file: .env
```

### 启用日志轮转

在生产环境配置 Docker 日志驱动:

```yaml
services:
  api-server:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### 资源限制

限制容器资源使用:

```yaml
services:
  api-server:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
```

## 监控与日志

### 查看日志

```bash
# 实时日志
docker compose logs -f

# 最近100行
docker compose logs --tail=100

# 特定时间段
docker compose logs --since 30m

# 导出日志
docker compose logs > app.log
```

### 容器指标

```bash
# 资源使用情况
docker stats z-ai-api-server

# 容器详情
docker inspect z-ai-api-server
```

## 安全建议

1. **修改默认密码**: 更改 `ADMIN_PASSWORD` 和 `AUTH_TOKEN`
2. **限制网络访问**: 生产环境使用反向代理 (Nginx/Caddy)
3. **启用 HTTPS**: 配置 SSL 证书
4. **定期备份**: 自动化数据库备份任务
5. **日志审计**: 定期检查 `request_logs` 表
6. **最小权限**: 避免以 root 运行容器

## 参考资料

- [Docker Compose 文档](https://docs.docker.com/compose/)
- [项目主 README](../README.md)
- [配置示例](.env.example)
