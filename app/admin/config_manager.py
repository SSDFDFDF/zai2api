"""Admin config metadata and helpers for the configuration console."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from dotenv import dotenv_values

from app.core.config import settings
from app.utils.env_file import update_env_file
from app.utils.logger import logger

ENV_PATH = Path(".env")
ENV_EXAMPLE_PATH = Path(".env.example")
_ENV_SOURCE_LINE_PATTERN = re.compile(
    r"^\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=.*$"
)


@dataclass(frozen=True)
class ConfigFieldSpec:
    key: str
    label: str
    description: str
    value_type: str
    default_value: object
    input_type: str = "text"
    placeholder: str = ""
    required: bool = False
    wide: bool = False
    sensitive: bool = False
    restart_required: bool = False
    min_value: int | None = None
    max_value: int | None = None
    db_persist: bool = True
    options: tuple[str, ...] = ()  # 仅 input_type="select" 时使用


@dataclass(frozen=True)
class ConfigSectionSpec:
    id: str
    title: str
    description: str
    fields: tuple[ConfigFieldSpec, ...]
    icon: str = ""


CONFIG_SECTIONS: tuple[ConfigSectionSpec, ...] = (
    ConfigSectionSpec(
        id="access",
        title="接入与认证",
        description="控制上游接口地址、客户端鉴权和 Function Call 行为。",
        icon="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z",
        fields=(
            ConfigFieldSpec(
                key="API_ENDPOINT",
                label="上游 API 地址",
                description="代理请求实际转发到的上游聊天完成接口。",
                value_type="str",
                default_value="https://chat.z.ai/api/v2/chat/completions",
                input_type="url",
                placeholder="https://chat.z.ai/api/v2/chat/completions",
                required=True,
                wide=True,
            ),
            ConfigFieldSpec(
                key="AUTH_TOKEN",
                label="客户端认证密钥",
                description="客户端访问本服务时使用的 Bearer Token。",
                value_type="str",
                default_value="sk-your-api-key",
                input_type="password",
                placeholder="sk-your-api-key",
                wide=True,
                sensitive=True,
            ),
            ConfigFieldSpec(
                key="SKIP_AUTH_TOKEN",
                label="跳过客户端认证",
                description="仅建议开发环境使用，开启后不校验 AUTH_TOKEN。",
                value_type="bool",
                default_value=False,
            ),
            ConfigFieldSpec(
                key="TOOL_STRATEGY",
                label="工具调用策略",
                description="xmlfc=XML提示词注入（默认）| native=原生透传 | glmnative=GLM原生 | disabled=禁用",
                value_type="str",
                default_value="xmlfc",
                input_type="select",
                options=("xmlfc", "native", "glmnative", "disabled"),
            ),
            ConfigFieldSpec(
                key="SCAN_LIMIT",
                label="工具调用扫描限制",
                description="Function Call 扫描的最大字符数。",
                value_type="int",
                default_value=200000,
                input_type="number",
                min_value=1,
                placeholder="200000",
            ),
            ConfigFieldSpec(
                key="TOOL_RESULT_MAX_LENGTH",
                label="工具结果截断长度",
                description="工具执行结果超过此长度时截断，0 表示不截断。",
                value_type="int",
                default_value=0,
                input_type="number",
                min_value=0,
                placeholder="0",
            ),
            ConfigFieldSpec(
                key="UPSTREAM_BACKGROUND_TASKS",
                label="上游背景任务",
                description="是否允许上游执行标题生成、搜索总结等背景任务（开启可能消耗更多额度/时间）。",
                value_type="bool",
                default_value=False,
            ),
            ConfigFieldSpec(
                key="SESSION_ENABLED",
                label="会话管理",
                description="启用官方链式会话复用，关闭则每次请求使用独立会话。",
                value_type="bool",
                default_value=False,
            ),
            ConfigFieldSpec(
                key="SESSION_SYSTEM_INJECT",
                label="System 注入模式",
                description="开启时将系统提示词注入到用户消息中（适用于链式会话和独立会话），关闭时作为独立 system 消息透传。",
                value_type="bool",
                default_value=False,
            ),
            ConfigFieldSpec(
                key="PRECREATE_CHAT",
                label="预创建 Chat ID",
                description="直接模式下通过上游接口预创建 chat_id，而非使用随机 UUID。仅在关闭会话管理时生效。",
                value_type="bool",
                default_value=False,
            ),
        ),
    ),
    ConfigSectionSpec(
        id="server",
        title="服务运行",
        description="服务监听、日志、数据库路径和反向代理前缀。",
        fields=(
            ConfigFieldSpec(
                key="SERVICE_NAME",
                label="服务名称",
                description="显示在进程列表中的服务名称。",
                value_type="str",
                default_value="api-proxy-server",
                placeholder="api-proxy-server",
                required=True,
                restart_required=True,
                db_persist=False,
            ),
            ConfigFieldSpec(
                key="LISTEN_PORT",
                label="监听端口",
                description="HTTP 服务监听端口。",
                value_type="int",
                default_value=8080,
                input_type="number",
                min_value=1,
                max_value=65535,
                required=True,
                restart_required=True,
                placeholder="8080",
                db_persist=False,
            ),
            ConfigFieldSpec(
                key="ROOT_PATH",
                label="反向代理路径前缀",
                description="例如 /api，部署在子路径时使用。",
                value_type="str",
                default_value="",
                placeholder="/api",
                restart_required=True,
                db_persist=False,
            ),
            ConfigFieldSpec(
                key="DEBUG_LOGGING",
                label="启用调试日志",
                description="开启后会输出更详细的调试信息。",
                value_type="bool",
                default_value=False,
                db_persist=False,
            ),
            ConfigFieldSpec(
                key="DB_PATH",
                label="数据库路径",
                description="SQLite 数据库文件位置。",
                value_type="str",
                default_value="tokens.db",
                placeholder="tokens.db",
                required=True,
                wide=True,
                restart_required=True,
                db_persist=False,
            ),
            ConfigFieldSpec(
                key="DATABASE_URL",
                label="PostgreSQL 连接串",
                description="（可选）如果使用 PostgreSQL，请填写形如 postgresql://user:pass@localhost:5432/dbname 的地址。留空则使用 SQLite。",
                value_type="str",
                default_value="",
                placeholder="postgresql://user:pass@localhost:5432/dbname",
                wide=True,
                restart_required=True,
                db_persist=False,
            ),
            ConfigFieldSpec(
                key="MAX_UPLOAD_FILE_SIZE",
                label="文件上传限制 (Bytes)",
                description="允许上传到上游的最大文件大小（默认 10MB）。",
                value_type="int",
                default_value=10485760,
                input_type="number",
                min_value=0,
                placeholder="10485760",
            ),
        ),
    ),
    ConfigSectionSpec(
        id="tokens",
        title="Token 池策略",
        description="失败判定、恢复时间和自动导入、自动维护计划任务。",
        icon="M4 7v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2H6c-1.1 0-2 .9-2 2zm0 4h16m-16 4h16",
        fields=(
            ConfigFieldSpec(
                key="TOKEN_FAILURE_THRESHOLD",
                label="失败阈值",
                description="连续失败多少次后将 Token 标记为不可用。",
                value_type="int",
                default_value=3,
                input_type="number",
                min_value=1,
                required=True,
                restart_required=True,
            ),
            ConfigFieldSpec(
                key="TOKEN_RECOVERY_TIMEOUT",
                label="恢复超时（秒）",
                description="失败 Token 重新参与调度前的等待时间。",
                value_type="int",
                default_value=1800,
                input_type="number",
                min_value=1,
                required=True,
                restart_required=True,
            ),
            ConfigFieldSpec(
                key="TOKEN_AUTO_IMPORT_ENABLED",
                label="启用自动导入",
                description="按固定周期扫描服务端目录并导入 Token。",
                value_type="bool",
                default_value=False,
            ),
            ConfigFieldSpec(
                key="TOKEN_AUTO_IMPORT_SOURCE_DIR",
                label="自动导入目录",
                description="服务端本地目录，开启自动导入时需要可访问。",
                value_type="str",
                default_value="",
                placeholder="E:\\tokens\\input",
                wide=True,
            ),
            ConfigFieldSpec(
                key="TOKEN_AUTO_IMPORT_INTERVAL",
                label="自动导入间隔（秒）",
                description="自动导入的扫描周期。",
                value_type="int",
                default_value=300,
                input_type="number",
                min_value=1,
                required=True,
            ),
            ConfigFieldSpec(
                key="TOKEN_AUTO_MAINTENANCE_ENABLED",
                label="启用自动维护",
                description="定时执行去重、健康检查和删除失效 Token。",
                value_type="bool",
                default_value=False,
            ),
            ConfigFieldSpec(
                key="TOKEN_AUTO_MAINTENANCE_INTERVAL",
                label="自动维护间隔（秒）",
                description="自动维护的执行周期。",
                value_type="int",
                default_value=1800,
                input_type="number",
                min_value=1,
                required=True,
            ),
            ConfigFieldSpec(
                key="TOKEN_AUTO_REMOVE_DUPLICATES",
                label="自动去重",
                description="自动维护时清理重复 Token。",
                value_type="bool",
                default_value=True,
            ),
            ConfigFieldSpec(
                key="TOKEN_AUTO_HEALTH_CHECK",
                label="自动健康检查",
                description="自动维护时验证 Token 可用性。",
                value_type="bool",
                default_value=True,
            ),
            ConfigFieldSpec(
                key="TOKEN_AUTO_DELETE_INVALID",
                label="自动删除失效 Token",
                description="自动维护时移除已验证为无效的 Token。",
                value_type="bool",
                default_value=False,
            ),
        ),
    ),
    ConfigSectionSpec(
        id="guest",
        title="匿名 Guest 会话池",
        description="没有用户 Token 时，控制匿名会话池的容量和维护策略。",
        icon="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z",
        fields=(
            ConfigFieldSpec(
                key="ANONYMOUS_MODE",
                label="启用匿名模式",
                description="无可用用户 Token 时允许使用匿名会话。",
                value_type="bool",
                default_value=True,
                restart_required=True,
            ),
            ConfigFieldSpec(
                key="GUEST_POOL_SIZE",
                label="Guest 池容量",
                description="启动和维持的 guest 会话数量。",
                value_type="int",
                default_value=3,
                input_type="number",
                min_value=1,
                required=True,
                restart_required=True,
            ),
            ConfigFieldSpec(
                key="GUEST_SESSION_MAX_AGE",
                label="Guest 最大存活时间（秒）",
                description="单个 guest 会话的最长存活时长。",
                value_type="int",
                default_value=480,
                input_type="number",
                min_value=1,
                required=True,
                restart_required=True,
            ),
            ConfigFieldSpec(
                key="GUEST_POOL_MAINTENANCE_INTERVAL",
                label="Guest 维护间隔（秒）",
                description="后台维护匿名会话池的频率。",
                value_type="int",
                default_value=30,
                input_type="number",
                min_value=1,
                required=True,
                restart_required=True,
            ),
            ConfigFieldSpec(
                key="GUEST_CLEANUP_PARALLELISM",
                label="清理并行度",
                description="并行回收空闲 guest 会话的数量。",
                value_type="int",
                default_value=4,
                input_type="number",
                min_value=1,
                required=True,
                restart_required=True,
            ),
            ConfigFieldSpec(
                key="GUEST_HTTP_MAX_KEEPALIVE_CONNECTIONS",
                label="Keep-Alive 连接数",
                description="guest 会话 HTTP 连接池 keep-alive 上限。",
                value_type="int",
                default_value=20,
                input_type="number",
                min_value=1,
                required=True,
                restart_required=True,
            ),
            ConfigFieldSpec(
                key="GUEST_HTTP_MAX_CONNECTIONS",
                label="HTTP 总连接数",
                description="guest 会话 HTTP 连接池的总连接数上限。",
                value_type="int",
                default_value=50,
                input_type="number",
                min_value=1,
                required=True,
                restart_required=True,
            ),
        ),
    ),
    ConfigSectionSpec(
        id="models",
        title="模型设置",
        description="在线模型自动解析与变体生成。模型列表从上游拉取后自动生成 Thinking / Search / Agent / advanced-search 变体。",
        icon="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4",
        fields=(
            ConfigFieldSpec(
                key="MODEL_BLACKLIST",
                label="模型黑名单",
                description="逗号分隔的上游模型 ID 列表，解析在线模型时跳过这些模型。例如：glm-4-flash,test-model",
                value_type="str",
                default_value="glm-4-flash",
                placeholder="glm-4-flash",
                wide=True,
            ),
            ConfigFieldSpec(
                key="MODEL_ALIASES",
                label="模型别名映射",
                description="自定义别名，让客户端可以用自定义名称访问模型。格式：别名=目标模型名，逗号分隔。例如：gpt-4o=GLM-5,gpt-4o-mini=GLM-4.7",
                value_type="str",
                default_value="",
                placeholder="gpt-4o=GLM-5,gpt-4o-mini=GLM-4.7",
                wide=True,
            ),
            ConfigFieldSpec(
                key="MODEL_AUTO_REFRESH_HOURS",
                label="自动刷新周期（小时）",
                description="定时从上游拉取在线模型列表并重新生成变体。设为 0 表示禁用自动刷新，仅手动刷新。",
                value_type="int",
                input_type="number",
                default_value=0,
                placeholder="0",
                min_value=0,
                max_value=168,
                restart_required=True,
            ),
            ConfigFieldSpec(
                key="GLM_INTERNAL_TOOL_HINT_ENABLED",
                label="启用 GLM 内部工具提示",
                description="向客户端展示模型后台隐式调用的工具（如 [联网检索]、[打开网页]）过程和完成状态。",
                value_type="bool",
                default_value=False,
                wide=True,
            ),
        ),
    ),
    ConfigSectionSpec(
        id="proxy",
        title="网络代理",
        description="用于所有上游访问的全局网络代理配置。",
        icon="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9",
        fields=(
            ConfigFieldSpec(
                key="HTTP_PROXY",
                label="代理服务器地址",
                description="统一代理，支持 http://, https://, socks5:// 等格式。例如：http://127.0.0.1:10808 或 socks5://127.0.0.1:1080。",
                value_type="str",
                default_value="",
                placeholder="http://127.0.0.1:10808",
                wide=True,
            ),
        ),
    ),
    ConfigSectionSpec(
        id="timeouts",
        title="超时控制",
        description="系统各环节的超时时间（单位：秒）。支持小数，修改即时生效。",
        icon="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z",
        fields=(
            ConfigFieldSpec(
                key="HTTP_CONNECT_TIMEOUT",
                label="连接超时",
                description="建立 TCP/TLS 连接的最大时间。",
                value_type="float",
                default_value=5.0,
                input_type="number",
                min_value=0.1,
                placeholder="5.0",
                db_persist=False,
            ),
            ConfigFieldSpec(
                key="HTTP_WRITE_TIMEOUT",
                label="写入超时",
                description="向网络套接字写入请求数据的最大时间。",
                value_type="float",
                default_value=10.0,
                input_type="number",
                min_value=0.1,
                placeholder="10.0",
                db_persist=False,
            ),
            ConfigFieldSpec(
                key="HTTP_POOL_TIMEOUT",
                label="连接池等待超时",
                description="从 HTTP 连接池获取可用连接的最大等待时间。",
                value_type="float",
                default_value=5.0,
                input_type="number",
                min_value=0.1,
                placeholder="5.0",
                db_persist=False,
            ),
            ConfigFieldSpec(
                key="HTTP_DEFAULT_READ_TIMEOUT",
                label="非流式读取超时",
                description="普通请求（如鉴权、获取模型列表）的单次读取超时时间。",
                value_type="float",
                default_value=60.0,
                input_type="number",
                min_value=1.0,
                placeholder="60.0",
                db_persist=False,
            ),
            ConfigFieldSpec(
                key="HTTP_STREAM_READ_TIMEOUT",
                label="流式单步读取超时",
                description="流式聊天中，相邻两个数据块(chunk)之间允许的最大静默时间。",
                value_type="float",
                default_value=120.0,
                input_type="number",
                min_value=1.0,
                placeholder="120.0",
                db_persist=False,
            ),
            ConfigFieldSpec(
                key="HTTP_MODEL_FETCH_TIMEOUT",
                label="模型拉取超时",
                description="请求上游 /v1/models 接口的超时时间。",
                value_type="float",
                default_value=10.0,
                input_type="number",
                min_value=1.0,
                placeholder="10.0",
                db_persist=False,
            ),
            ConfigFieldSpec(
                key="CHAT_TOTAL_TIMEOUT",
                label="非流式端到端总超时",
                description="单次非流式聊天请求的总处理时间上限（包含所有内部重试）。",
                value_type="float",
                default_value=300.0,
                input_type="number",
                min_value=1.0,
                placeholder="300.0",
                db_persist=False,
                wide=True,
            ),
            ConfigFieldSpec(
                key="HTTP_STREAM_TOTAL_TIMEOUT",
                label="流式端到端总超时",
                description="单次流式聊天请求的总处理时间上限（包含首包等待及所有抛出数据的总耗时）。",
                value_type="float",
                default_value=600.0,
                input_type="number",
                min_value=1.0,
                placeholder="600.0",
                db_persist=False,
                wide=True,
            ),
        ),
    ),
    ConfigSectionSpec(
        id="admin",
        title="后台安全",
        description="管理后台密码和会话密钥。修改后建议重新登录。",
        icon="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z",
        fields=(
            ConfigFieldSpec(
                key="ADMIN_PASSWORD",
                label="后台密码",
                description="管理后台登录密码。",
                value_type="str",
                default_value="admin123",
                input_type="password",
                placeholder="admin123",
                required=True,
                sensitive=True,
            ),
            ConfigFieldSpec(
                key="SESSION_SECRET_KEY",
                label="会话密钥",
                description="用于后台会话签名的密钥。",
                value_type="str",
                default_value="your-secret-key-change-in-production",
                input_type="password",
                placeholder="your-secret-key-change-in-production",
                required=True,
                sensitive=True,
                wide=True,
            ),
        ),
    ),
    ConfigSectionSpec(
        id="personalization",
        title="上游个性化",
        description="配置发送给上游的用户画像变量，用于自定义模型回复风格。",
        icon="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z",
        fields=(
            ConfigFieldSpec(
                key="UPSTREAM_USER_NAME",
                label="用户名称",
                description="上游上下文中的 {{USER_NAME}} 变量。",
                value_type="str",
                default_value="Guest",
            ),
            ConfigFieldSpec(
                key="UPSTREAM_USER_LOCATION",
                label="用户位置",
                description="上游上下文中的 {{USER_LOCATION}} 变量。",
                value_type="str",
                default_value="Unknown",
            ),
            ConfigFieldSpec(
                key="UPSTREAM_USER_TIMEZONE",
                label="用户时区",
                description="上游上下文中的 {{CURRENT_TIMEZONE}} 变量。",
                value_type="str",
                default_value="Asia/Shanghai",
            ),
            ConfigFieldSpec(
                key="UPSTREAM_USER_LANGUAGE",
                label="用户语言",
                description="上游上下文中的 {{USER_LANGUAGE}} 变量。",
                value_type="str",
                default_value="zh-CN",
            ),
        ),
    ),
)

CONFIG_FIELD_SPECS = {
    field.key: field
    for section in CONFIG_SECTIONS
    for field in section.fields
}
MANAGED_ENV_KEYS = tuple(CONFIG_FIELD_SPECS.keys())
DB_PERSIST_KEYS = frozenset(
    key for key, field in CONFIG_FIELD_SPECS.items() if field.db_persist
)
ReloadCallback = Callable[[], Awaitable[None]]


def read_env_content(env_path: str | Path = ENV_PATH) -> str:
    path = Path(env_path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def validate_env_source(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")

    for line_number, line in enumerate(normalized.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not _ENV_SOURCE_LINE_PATTERN.match(line):
            raise ValueError(
                f"第 {line_number} 行不是合法的 KEY=VALUE 格式。"
            )

    return normalized


def _serialize_config_value(value: object) -> str:
    """将 Python 值序列化为数据库存储的字符串。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _parse_db_value(raw: str, value_type: str) -> Any:
    """按字段类型解析数据库中的字符串值。"""
    if value_type == "bool":
        return raw.lower() in ("true", "1", "yes")
    if value_type == "int":
        try:
            return int(raw)
        except (ValueError, TypeError):
            return None
    if value_type == "float":
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None
    return raw


async def load_db_overrides() -> dict[str, Any]:
    """从数据库加载可持久化的配置值，按类型解析后返回。

    仅返回解析成功且属于 DB_PERSIST_KEYS 的条目。
    如果数据库不可用则静默返回空字典。
    """
    try:
        from app.services.config_dao import get_config_dao

        dao = get_config_dao()
        raw_items = await dao.get_all()
    except Exception as exc:
        logger.debug(f"⚠️ 读取数据库配置失败，回退到环境变量: {exc}")
        return {}

    overrides: dict[str, Any] = {}
    for key, raw_value in raw_items.items():
        if key not in DB_PERSIST_KEYS:
            continue
        field = CONFIG_FIELD_SPECS.get(key)
        if field is None:
            continue
        parsed = _parse_db_value(raw_value, field.value_type)
        if parsed is not None or field.value_type == "str":
            overrides[key] = parsed
    return overrides


async def apply_db_overrides(settings_obj: Any = settings) -> None:
    """从数据库加载配置并覆盖到 settings 对象上。"""
    overrides = await load_db_overrides()
    for key, value in overrides.items():
        setattr(settings_obj, key, value)
    if overrides:
        logger.info(f"🗄️ 已从数据库加载 {len(overrides)} 项配置覆盖")


def build_config_page_data(
    *,
    settings_obj: Any = settings,
    env_path: str | Path = ENV_PATH,
    env_example_path: str | Path = ENV_EXAMPLE_PATH,
    db_values: dict[str, str] | None = None,
) -> dict[str, Any]:
    env_file = Path(env_path)
    env_content = read_env_content(env_file)
    env_values = dotenv_values(env_file) if env_file.exists() else {}
    db_keys = set(db_values.keys()) if db_values else set()
    sections: list[dict[str, Any]] = []
    total_fields = 0
    overridden_fields = 0
    sensitive_fields = 0
    restart_required_fields = 0

    for section in CONFIG_SECTIONS:
        rendered_fields: list[dict[str, Any]] = []
        for field in section.fields:
            total_fields += 1
            if field.sensitive:
                sensitive_fields += 1
            if field.restart_required:
                restart_required_fields += 1

            # 判断来源优先级：数据库 > .env > 默认值
            in_db = field.key in db_keys
            in_env = field.key in env_values
            if in_db:
                source_label = "数据库"
                source_badge_class = (
                    "bg-blue-50 text-blue-700 ring-blue-200"
                )
                overridden_fields += 1
            elif in_env:
                source_label = ".env"
                source_badge_class = (
                    "bg-emerald-50 text-emerald-700 ring-emerald-200"
                )
                overridden_fields += 1
            else:
                source_label = "默认值"
                source_badge_class = (
                    "bg-slate-100 text-slate-600 ring-slate-200"
                )

            value = getattr(settings_obj, field.key, field.default_value)
            if value is None:
                value = ""

            rendered_fields.append(
                {
                    "key": field.key,
                    "label": field.label,
                    "description": field.description,
                    "value_type": field.value_type,
                    "value": value,
                    "input_type": field.input_type,
                    "placeholder": field.placeholder,
                    "required": field.required,
                    "wide": field.wide,
                    "sensitive": field.sensitive,
                    "restart_required": field.restart_required,
                    "min_value": field.min_value,
                    "max_value": field.max_value,
                    "source_label": source_label,
                    "source_badge_class": source_badge_class,
                    "options": field.options,
                }
            )

        sections.append(
            {
                "id": section.id,
                "title": section.title,
                "description": section.description,
                "icon": section.icon,
                "fields": rendered_fields,
                "field_count": len(rendered_fields),
            }
        )

    return {
        "sections": sections,
        "env_content": env_content,
        "overview": {
            "total_sections": len(CONFIG_SECTIONS),
            "total_fields": total_fields,
            "overridden_fields": overridden_fields,
            "default_fields": total_fields - overridden_fields,
            "sensitive_fields": sensitive_fields,
            "restart_required_fields": restart_required_fields,
            "env_exists": env_file.exists(),
            "env_path": str(env_file.resolve()),
            "env_line_count": len(env_content.splitlines()) if env_content else 0,
            "example_exists": Path(env_example_path).exists(),
        },
    }


def build_form_updates(form_data: Mapping[str, Any]) -> dict[str, object]:
    updates: dict[str, object] = {}

    for key in MANAGED_ENV_KEYS:
        field = CONFIG_FIELD_SPECS[key]

        if field.value_type == "bool":
            updates[key] = key in form_data
            continue

        raw_value = str(form_data.get(key, "") or "").strip()
        if field.required and raw_value == "":
            raise ValueError(f"{field.label} 不能为空。")

        if field.value_type == "int":
            try:
                parsed = int(raw_value)
            except ValueError as exc:
                raise ValueError(f"{field.label} 必须是整数。") from exc

            if field.min_value is not None and parsed < field.min_value:
                raise ValueError(
                    f"{field.label} 不能小于 {field.min_value}。"
                )
            if field.max_value is not None and parsed > field.max_value:
                raise ValueError(
                    f"{field.label} 不能大于 {field.max_value}。"
                )
            updates[key] = parsed
            continue

        if field.value_type == "float":
            try:
                parsed_float = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"{field.label} 必须是数字。") from exc

            if field.min_value is not None and parsed_float < field.min_value:
                raise ValueError(
                    f"{field.label} 不能小于 {field.min_value}。"
                )
            if field.max_value is not None and parsed_float > field.max_value:
                raise ValueError(
                    f"{field.label} 不能大于 {field.max_value}。"
                )
            updates[key] = parsed_float
            continue

        updates[key] = raw_value

    return updates


async def _apply_env_change(
    writer: Callable[[Path], None],
    *,
    reload_callback: ReloadCallback,
    env_path: str | Path = ENV_PATH,
) -> None:
    path = Path(env_path)
    had_existing_file = path.exists()
    previous_content = read_env_content(path) if had_existing_file else ""

    try:
        writer(path)
        await reload_callback()
    except Exception:
        if had_existing_file:
            path.write_text(previous_content, encoding="utf-8")
        elif path.exists():
            path.unlink()

        try:
            await reload_callback()
        except Exception as restore_exc:
            logger.warning(f"⚠️ 回滚配置后重新加载失败: {restore_exc}")
        raise


async def save_form_config(
    form_data: Mapping[str, Any],
    *,
    reload_callback: ReloadCallback,
    env_path: str | Path = ENV_PATH,
) -> dict[str, object]:
    updates = build_form_updates(form_data)

    # 分流：db_persist=True → 数据库，其余 → .env
    db_updates: dict[str, str] = {}
    env_updates: dict[str, object] = {}
    for key, value in updates.items():
        if key in DB_PERSIST_KEYS:
            db_updates[key] = _serialize_config_value(value)
        else:
            env_updates[key] = value

    # 写入数据库
    if db_updates:
        try:
            from app.services.config_dao import get_config_dao

            dao = get_config_dao()
            await dao.set_many(db_updates)
            logger.info(f"🗄️ 已保存 {len(db_updates)} 项配置到数据库")
        except Exception as exc:
            logger.warning(
                f"⚠️ 数据库保存失败，回退到环境变量: {exc}"
            )
            env_updates.update(updates)  # fallback: 全部写入 .env
            db_updates.clear()

    # 写入 .env（仅非数据库字段或回退字段）
    if env_updates:
        async def _reload() -> None:
            await reload_callback()

        def _writer(target_path: Path) -> None:
            update_env_file(env_updates, env_path=target_path)

        await _apply_env_change(
            _writer, reload_callback=_reload, env_path=env_path
        )
    else:
        # 即使没有 env 变更，也需要重新加载以应用 DB 值
        await reload_callback()

    return updates


async def save_source_config(
    env_content: str,
    *,
    reload_callback: ReloadCallback,
    env_path: str | Path = ENV_PATH,
) -> None:
    normalized = validate_env_source(env_content)

    def _writer(target_path: Path) -> None:
        content = normalized.rstrip("\n")
        target_path.write_text(
            f"{content}\n" if content else "",
            encoding="utf-8",
        )

    await _apply_env_change(
        _writer,
        reload_callback=reload_callback,
        env_path=env_path,
    )


async def reset_env_to_example(
    *,
    reload_callback: ReloadCallback,
    env_path: str | Path = ENV_PATH,
    env_example_path: str | Path = ENV_EXAMPLE_PATH,
) -> None:
    example_path = Path(env_example_path)
    if not example_path.exists():
        raise FileNotFoundError(".env.example 不存在")

    example_content = example_path.read_text(encoding="utf-8")

    def _writer(target_path: Path) -> None:
        content = example_content.rstrip("\n")
        target_path.write_text(
            f"{content}\n" if content else "",
            encoding="utf-8",
        )

    await _apply_env_change(
        _writer,
        reload_callback=reload_callback,
        env_path=env_path,
    )
