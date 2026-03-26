"""
管理后台认证中间件
"""
from fastapi import Request, HTTPException, status
from typing import Optional
import secrets
import base64
import hmac
import json
import time

from app.core.config import settings

# Session 有效期（小时）
SESSION_EXPIRE_HOURS = 24
SESSION_COOKIE_NAME = "admin_session"
CSRF_COOKIE_NAME = "admin_csrf"
CSRF_HEADER_NAME = "X-CSRF-Token"

LOGIN_WINDOW_SECONDS = 300
LOGIN_MAX_ATTEMPTS = 5
_login_attempts: dict[str, list[float]] = {}

# ── Session 吊销黑名单（进程内存，重启后清空）─────────────────────────────
# 键: session token 原始字符串, 值: 吊销时间戳
_revoked_sessions: dict[str, float] = {}


def _sign_payload(payload: str) -> str:
    secret = settings.SESSION_SECRET_KEY or ""
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), "sha256").hexdigest()


def _encode_session(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    b64 = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
    signature = _sign_payload(b64)
    return f"{b64}.{signature}"


def _decode_session(token: str) -> Optional[dict]:
    if not token or "." not in token:
        return None
    b64, signature = token.split(".", 1)
    expected = _sign_payload(b64)
    if not hmac.compare_digest(signature, expected):
        return None
    padded = b64 + "=" * (-len(b64) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("utf-8"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _now() -> float:
    return time.time()


def _prune_attempts(attempts: list[float]) -> list[float]:
    cutoff = _now() - LOGIN_WINDOW_SECONDS
    return [ts for ts in attempts if ts >= cutoff]


def check_login_rate_limit(client_id: str) -> bool:
    attempts = _login_attempts.get(client_id, [])
    attempts = _prune_attempts(attempts)
    if attempts:
        _login_attempts[client_id] = attempts
    else:
        _login_attempts.pop(client_id, None)
    return len(attempts) < LOGIN_MAX_ATTEMPTS


def record_login_failure(client_id: str) -> None:
    attempts = _login_attempts.get(client_id, [])
    attempts = _prune_attempts(attempts)
    attempts.append(_now())
    _login_attempts[client_id] = attempts


def reset_login_failures(client_id: str) -> None:
    _login_attempts.pop(client_id, None)


def create_session(password: str) -> Optional[str]:
    """
    创建 session

    Args:
        password: 用户输入的密码

    Returns:
        session_token 或 None（密码错误）
    """
    # 常数时间密码比较，防止时序侧信道
    if not hmac.compare_digest(password.encode("utf-8"), settings.ADMIN_PASSWORD.encode("utf-8")):
        return None

    issued_at = int(_now())
    expires_at = int((_now() + SESSION_EXPIRE_HOURS * 3600))
    payload = {
        "iat": issued_at,
        "exp": expires_at,
        "auth": True,
        # 每个 session 携带唯一 jti，确保吊销时能精确匹配
        "jti": secrets.token_hex(16),
    }
    return _encode_session(payload)


def verify_session(session_token: Optional[str]) -> bool:
    """
    验证 session 是否有效

    Args:
        session_token: Session token

    Returns:
        是否已认证
    """
    if not session_token:
        return False

    # 检查吊销黑名单
    if session_token in _revoked_sessions:
        return False

    session = _decode_session(session_token)
    if not session:
        return False

    exp = session.get("exp")
    if isinstance(exp, (int, float)) and _now() > float(exp):
        return False

    return bool(session.get("auth"))


def delete_session(session_token: Optional[str]) -> None:
    """将 session token 加入吊销黑名单（真正的登出）"""
    if session_token:
        _revoked_sessions[session_token] = _now()
        _prune_revoked_sessions()


def _prune_revoked_sessions() -> None:
    """清理已过期的黑名单条目（避免内存无限增长）"""
    cutoff = _now() - SESSION_EXPIRE_HOURS * 3600
    expired_keys = [k for k, ts in _revoked_sessions.items() if ts < cutoff]
    for k in expired_keys:
        del _revoked_sessions[k]


def get_session_token_from_request(request: Request) -> Optional[str]:
    """从请求中获取 session token"""
    return request.cookies.get(SESSION_COOKIE_NAME)


# ── CSRF: 绑定 Session 的 Double Submit Cookie ────────────────────────────
# 格式: <random_part>.<hmac_tag>
# hmac_tag = HMAC-SHA256(secret, random_part + ":" + session_token)[0:16]
# 这样即使攻击者能注入任意 Cookie，也无法伪造与当前 Session 绑定的 CSRF token

def generate_csrf_token(session_token: str) -> str:
    """生成与 Session 密码学绑定的 CSRF token"""
    random_part = secrets.token_urlsafe(32)
    secret = settings.SESSION_SECRET_KEY or ""
    tag = hmac.new(
        secret.encode("utf-8"),
        f"{random_part}:{session_token}".encode("utf-8"),
        "sha256",
    ).hexdigest()[:16]
    return f"{random_part}.{tag}"


def verify_csrf_token(csrf_token: str, session_token: str) -> bool:
    """校验 CSRF token 是否与当前 Session 匹配"""
    if not csrf_token or "." not in csrf_token:
        return False
    random_part, tag = csrf_token.rsplit(".", 1)
    secret = settings.SESSION_SECRET_KEY or ""
    expected_tag = hmac.new(
        secret.encode("utf-8"),
        f"{random_part}:{session_token}".encode("utf-8"),
        "sha256",
    ).hexdigest()[:16]
    return hmac.compare_digest(tag, expected_tag)


def get_csrf_token_from_request(request: Request) -> Optional[str]:
    """从请求中获取 CSRF token（Cookie）"""
    return request.cookies.get(CSRF_COOKIE_NAME)


def get_csrf_header_from_request(request: Request) -> Optional[str]:
    return request.headers.get(CSRF_HEADER_NAME)


async def require_auth(request: Request):
    """
    认证依赖项：要求用户已登录

    在路由中使用：
    @router.get("/admin", dependencies=[Depends(require_auth)])
    """
    session_token = get_session_token_from_request(request)

    if not verify_session(session_token):
        # 未认证，重定向到登录页
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            detail="未登录",
            headers={"Location": "/admin/login"}
        )


async def require_csrf(request: Request):
    """CSRF 保护，仅针对非安全方法（POST/PUT/DELETE/PATCH）"""
    if request.method.upper() in {"GET", "HEAD", "OPTIONS"}:
        return

    session_token = get_session_token_from_request(request)
    csrf_cookie = get_csrf_token_from_request(request)
    csrf_header = get_csrf_header_from_request(request)

    # header 必须与 cookie 一致（Double Submit）
    if not csrf_cookie or not csrf_header or not hmac.compare_digest(csrf_cookie, csrf_header):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid CSRF token",
        )

    # 进一步验证 token 是否与当前 Session 绑定（防 Cookie 注入绕过）
    if not session_token or not verify_csrf_token(csrf_cookie, session_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid CSRF token",
        )


def get_authenticated_user(request: Request) -> bool:
    """
    获取当前认证状态（用于模板）

    Returns:
        是否已认证
    """
    session_token = get_session_token_from_request(request)
    return verify_session(session_token)


def cleanup_expired_sessions():
    """清理过期的黑名单条目（定时任务调用）"""
    _prune_revoked_sessions()
    return len(_revoked_sessions)
