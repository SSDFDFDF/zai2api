# Nginx 反向代理部署指南

本文档说明如何在 Nginx 反向代理后部署 Z.AI2API,支持自定义路径前缀。

## 问题说明

在使用 Nginx 反向代理时,如果需要将服务部署在自定义路径前缀下(例如 `http://domain.com/ai2api`),
需要正确配置 `ROOT_PATH` 环境变量,否则会出现以下问题:

- 后台管理页面跳转错误(缺少路径前缀)
- API 接口请求 404(路径不完整)
- 静态资源加载失败

## 解决方案

### 1. 配置环境变量

在 `.env` 文件中设置 `ROOT_PATH` 变量,值为 Nginx 配置的 location 路径:

```bash
# 示例:部署在 /ai2api 路径下
ROOT_PATH=/ai2api
```

**重要**: `ROOT_PATH` 必须与 Nginx 配置中的 `location` 路径完全一致。

### 2. 配置 Nginx

参考 `deploy/nginx.conf.example` 文件,选择合适的配置模板。

#### 基础配置示例

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /ai2api {
        # 代理到后端服务
        proxy_pass http://127.0.0.1:8080;

        # 传递原始请求信息
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE 流式响应支持
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_buffering off;
        proxy_cache off;

        # 超时设置
        proxy_read_timeout 300s;
    }
}
```

### 3. Docker Compose 配置

如果使用 Docker 部署,需要在 `docker-compose.yml` 中添加 `ROOT_PATH` 环境变量:

```yaml
version: '3.8'
services:
  ai2api:
    image: z-ai2api:latest
    environment:
      - ROOT_PATH=/ai2api
      - LISTEN_PORT=8080
      # ... 其他环境变量
    ports:
      - "8080:8080"
```

### 4. 重启服务

```bash
# 重载 Nginx 配置
sudo nginx -t
sudo systemctl reload nginx

# 重启应用(Docker)
docker-compose restart

# 或重启应用(直接运行)
# 停止服务后重新启动
```

## 访问地址

配置完成后,服务访问地址如下:

- **API 端点**: `http://your-domain.com/ai2api/v1/chat/completions`
- **模型列表**: `http://your-domain.com/ai2api/v1/models`
- **管理后台**: `http://your-domain.com/ai2api/admin/login`
- **根路径**: `http://your-domain.com/ai2api/`

## 配置示例

### 示例 1: 部署在 /api 路径下

**.env 配置**:
```bash
ROOT_PATH=/api
```

**Nginx 配置**:
```nginx
location /api {
    proxy_pass http://127.0.0.1:8080;
    # ... 其他配置
}
```

**访问地址**: `http://domain.com/api/admin/login`

### 示例 2: 部署在根路径(无前缀)

**.env 配置**:
```bash
ROOT_PATH=
```

**Nginx 配置**:
```nginx
location / {
    proxy_pass http://127.0.0.1:8080;
    # ... 其他配置
}
```

**访问地址**: `http://domain.com/admin/login`

### 示例 3: 多级路径前缀

**.env 配置**:
```bash
ROOT_PATH=/services/ai/chat
```

**Nginx 配置**:
```nginx
location /services/ai/chat {
    proxy_pass http://127.0.0.1:8080;
    # ... 其他配置
}
```

**访问地址**: `http://domain.com/services/ai/chat/admin/login`

## 常见问题排查

### 1. 404 错误

**现象**: 访问页面或 API 时返回 404

**可能原因**:
- `ROOT_PATH` 配置与 Nginx location 路径不匹配
- Nginx 配置错误或未重载

**解决方法**:
- 检查 `.env` 中的 `ROOT_PATH` 是否与 Nginx `location` 完全一致
- 确认 Nginx 配置无误: `sudo nginx -t`
- 重载 Nginx: `sudo systemctl reload nginx`
- 重启应用服务

### 2. 静态资源加载失败

**现象**: 管理后台页面样式错乱,控制台显示 CSS/JS 404

**可能原因**:
- `ROOT_PATH` 未配置或配置错误
- 静态文件路径未包含前缀

**解决方法**:
- 确保 `ROOT_PATH` 正确配置并重启服务
- 检查浏览器开发者工具中的资源请求路径

### 3. 流式响应中断

**现象**: SSE 流式响应提前终止或无法正常工作

**可能原因**:
- Nginx 启用了缓冲
- 超时时间设置过短

**解决方法**:
在 Nginx 配置中添加:
```nginx
proxy_buffering off;
proxy_cache off;
proxy_read_timeout 300s;
```

### 4. CORS 错误

**现象**: 浏览器控制台显示跨域请求被阻止

**可能原因**:
- Nginx 未正确传递请求头

**解决方法**:
确保 Nginx 配置中包含:
```nginx
proxy_set_header Host $host;
proxy_set_header X-Forwarded-Proto $scheme;
```

## 验证配置

配置完成后,可以通过以下方式验证:

1. **访问健康检查端点**:
   ```bash
   curl http://your-domain.com/ai2api/v1/models
   ```

2. **访问管理后台**:
   在浏览器打开 `http://your-domain.com/ai2api/admin/login`

3. **测试 API 请求**:
   ```bash
   curl -X POST http://your-domain.com/ai2api/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-api-key" \
     -d '{
       "model": "GLM-4.6",
       "messages": [{"role": "user", "content": "Hello"}],
       "stream": false
     }'
   ```

## 进阶配置

### HTTPS 配置

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location /ai2api {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header X-Forwarded-Proto https;
        # ... 其他配置
    }
}
```

### 负载均衡

```nginx
upstream ai2api_backend {
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
    server 127.0.0.1:8082;
}

server {
    listen 80;
    location /ai2api {
        proxy_pass http://ai2api_backend;
        # ... 其他配置
    }
}
```

## 参考资料

- [FastAPI Behind a Proxy](https://fastapi.tiangolo.com/advanced/behind-a-proxy/)
- [Nginx Proxy Module](http://nginx.org/en/docs/http/ngx_http_proxy_module.html)
- 完整配置示例: `deploy/nginx.conf.example`
