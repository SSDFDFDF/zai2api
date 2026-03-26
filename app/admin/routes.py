"""
管理后台路由模块
"""
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from app.admin.auth import require_auth
from app.admin.config_manager import build_config_page_data
from app.admin.stats import (
    DEFAULT_TREND_WINDOW,
    TREND_WINDOW_OPTIONS,
    collect_admin_stats,
    get_process_uptime,
)
from app.admin.template_loader import templates
from app.utils.logger import logger

router = APIRouter(prefix="/admin", tags=["admin"])
DEFAULT_TOKEN_NAMESPACE = "zai"


def _page_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }


def _base_context(request: Request) -> dict:
    return {
        "request": request,
        "root_path": request.scope.get("root_path", ""),
    }


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """登录页面"""
    try:
        return templates.TemplateResponse(
            "login.html",
            _base_context(request),
            headers=_page_headers(),
        )
    except Exception:
        logger.exception("渲染管理后台登录页失败")
        return HTMLResponse(
            "<h1>Admin login page failed to render</h1>",
            status_code=500,
            headers=_page_headers(),
        )


@router.get("/", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def dashboard(request: Request):
    """仪表盘首页"""
    try:
        stats = await collect_admin_stats(
            DEFAULT_TOKEN_NAMESPACE,
            trend_window=DEFAULT_TREND_WINDOW,
        )
        stats["uptime"] = get_process_uptime()

        context = {
            **_base_context(request),
            "stats": stats,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trend_windows": TREND_WINDOW_OPTIONS,
        }

        return templates.TemplateResponse("index.html", context, headers=_page_headers())
    except Exception:
        logger.exception("渲染管理后台仪表盘失败")
        raise


@router.get(
    "/config",
    response_class=HTMLResponse,
    dependencies=[Depends(require_auth)],
)
async def config_page(request: Request):
    """配置管理页面"""
    try:
        # 加载数据库中的配置值，用于展示来源标识
        try:
            from app.services.config_dao import get_config_dao

            dao = get_config_dao()
            db_values = await dao.get_all()
        except Exception:
            db_values = {}

        page_data = build_config_page_data(db_values=db_values)

        # 模型状态数据
        model_list = []
        model_parsed = False
        try:
            from app.core.openai import get_upstream_client_if_ready

            client = get_upstream_client_if_ready()
            if client:
                mgr = client._model_manager
                model_parsed = mgr._parsed
                for name in mgr.get_supported_models():
                    caps = mgr.get_model_capabilities(name)
                    upstream_id = mgr.get_upstream_model_id(name) or ""
                    mcp = mgr.get_mcp_servers(name)
                    scene = mgr.get_scene_defaults(name)
                    model_list.append({
                        "name": name,
                        "upstream_id": upstream_id,
                        "tool_use": caps.get("tool_use", False),
                        "vision": caps.get("vision", False),
                        "thinking": caps.get("thinking", False),
                        "web_search": caps.get("web_search", False),
                        "agent": caps.get("agent", False),
                        "mcp_servers": mcp,
                        "scene_defaults": scene,
                    })
        except Exception:
            pass

        context = {
            **_base_context(request),
            "sections": page_data["sections"],
            "env_content": page_data["env_content"],
            "overview": page_data["overview"],
            "model_list": model_list,
            "model_parsed": model_parsed,
            "model_count": len(model_list),
        }
        return templates.TemplateResponse("config.html", context, headers=_page_headers())
    except Exception:
        logger.exception("渲染管理后台配置页失败")
        raise


@router.get("/request-logs", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def request_logs_page(request: Request):
    """请求日志页面"""
    try:
        return templates.TemplateResponse(
            "request_logs.html",
            _base_context(request),
            headers=_page_headers(),
        )
    except Exception:
        logger.exception("渲染管理后台请求日志页失败")
        raise


@router.get(
    "/tokens",
    response_class=HTMLResponse,
    dependencies=[Depends(require_auth)],
)
async def tokens_page(request: Request):
    """Token 管理页面"""
    try:
        from app.core.config import settings

        maintenance_actions: list[str] = []
        if settings.TOKEN_AUTO_REMOVE_DUPLICATES:
            maintenance_actions.append("删除重复 Token")
        if settings.TOKEN_AUTO_HEALTH_CHECK:
            maintenance_actions.append("批量测活")
        if settings.TOKEN_AUTO_DELETE_INVALID:
            maintenance_actions.append("删除失效 Token")

        context = {
            **_base_context(request),
            "automation": {
                "config_url": f"{request.scope.get('root_path', '')}/admin/config#tokens",
                "import_enabled": settings.TOKEN_AUTO_IMPORT_ENABLED,
                "import_source_dir": settings.TOKEN_AUTO_IMPORT_SOURCE_DIR,
                "import_interval": settings.TOKEN_AUTO_IMPORT_INTERVAL,
                "has_import_source_dir": bool(
                    settings.TOKEN_AUTO_IMPORT_SOURCE_DIR.strip()
                ),
                "maintenance_enabled": settings.TOKEN_AUTO_MAINTENANCE_ENABLED,
                "maintenance_interval": settings.TOKEN_AUTO_MAINTENANCE_INTERVAL,
                "maintenance_actions": maintenance_actions,
                "has_maintenance_actions": bool(maintenance_actions),
            },
        }
        return templates.TemplateResponse("tokens.html", context, headers=_page_headers())
    except Exception:
        logger.exception("渲染管理后台 Token 页失败")
        raise
