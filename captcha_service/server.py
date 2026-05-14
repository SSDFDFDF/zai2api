#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Z.ai Captcha Token Service

Standalone FastAPI service that manages a Playwright browser session to
generate Aliyun Captcha tokens for Z.ai API requests.

Deploy separately from the main API proxy to isolate browser memory usage.

Endpoints:
    POST /token  - Generate a fresh captcha token
    GET  /health - Service health check
"""

import asyncio
import base64
import json
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger("captcha")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAPTCHA_HOST = os.getenv("CAPTCHA_HOST", "0.0.0.0")
CAPTCHA_PORT = int(os.getenv("CAPTCHA_PORT", "9000"))
CAPTCHA_HEADLESS = os.getenv("CAPTCHA_HEADLESS", "true").lower() == "true"
CAPTCHA_TOKEN_TIMEOUT = float(os.getenv("CAPTCHA_TOKEN_TIMEOUT", "15.0"))
CAPTCHA_PAGE_IDLE_TTL = float(os.getenv("CAPTCHA_PAGE_IDLE_TTL", "300"))


# Stealth init script — hide automation signals so Aliyun silent-verify doesn't
# fall back to the slider puzzle (which a headless browser can't solve).
_STEALTH_INIT_SCRIPT = r"""
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
window.chrome = { runtime: {}, loadTimes: function(){}, csi: function(){}, app: {} };
const _origPermQuery = window.navigator.permissions && window.navigator.permissions.query;
if (_origPermQuery) {
    window.navigator.permissions.query = (p) => p.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : _origPermQuery(p);
}
const _origGetParam = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(p) {
    if (p === 37445) return 'Intel Inc.';
    if (p === 37446) return 'Intel Iris OpenGL Engine';
    return _origGetParam.call(this, p);
};
"""


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def decode_jwt_payload(token: str) -> Dict:
    """Decode JWT payload without signature verification."""
    parts = token.split(".")
    if len(parts) < 2:
        raise ValueError("Invalid JWT: expected 3 dot-separated parts")
    padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
    return json.loads(base64.urlsafe_b64decode(padded))


def extract_user_id(jwt_token: str) -> str:
    payload = decode_jwt_payload(jwt_token)
    user_id = payload.get("id")
    if not user_id:
        raise ValueError("JWT missing 'id' field")
    return user_id


# ---------------------------------------------------------------------------
# Browser session manager
# ---------------------------------------------------------------------------

class CaptchaBrowser:
    """Manages a single Playwright browser with per-user page tabs."""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None
        self._pages: Dict[str, Dict] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=CAPTCHA_HEADLESS,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1265, "height": 1281},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
            ),
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
        )
        await self._context.add_init_script(_STEALTH_INIT_SCRIPT)
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(f"Browser started (headless={CAPTCHA_HEADLESS})")

        # 连通性自检
        await self._connectivity_check()

    async def _connectivity_check(self):
        """启动时验证 chat.z.ai 可达且 captcha SDK 可用。"""
        logger.info("running connectivity check...")
        page = await self._context.new_page()
        errors = []

        # 1. DNS / TCP 可达性 + 等待 Svelte 渲染完成
        try:
            resp = await page.goto(
                "https://chat.z.ai",
                wait_until="domcontentloaded",
                timeout=20000,
            )
            if resp is None:
                errors.append("page.goto returned None (possible network block)")
            elif resp.status >= 400:
                errors.append(f"HTTP {resp.status}")
            else:
                title = await page.title()
                logger.info(f"   page loaded: status={resp.status} title={title!r}")
        except Exception as e:
            errors.append(f"cannot reach chat.z.ai: {e}")
            logger.warning(f"  FAIL: page load error: {e}")
            await page.close()
            if errors:
                logger.warning("CONNECTIVITY CHECK FAILED:")
                for e in errors:
                    logger.warning(f"  - {e}")
            return

        # 2. 等待 Svelte 渲染 chat UI（DOMContentLoaded 后 Svelte 异步挂载）
        try:
            await page.wait_for_selector(
                'textarea, [contenteditable="true"], [role="textbox"]',
                timeout=15000,
            )
            logger.info(f"   input element appeared after wait")
        except Exception:
            logger.warning("   input element did not appear within 15s")

        # 3. Chat UI 完整性
        checks = await page.evaluate("""() => {
            const textarea = document.querySelector('textarea');
            const editableDiv = document.querySelector('[contenteditable="true"]');
            const roleTextbox = document.querySelector('[role="textbox"]');
            return {
                textarea: !!textarea,
                contentEditableTag: editableDiv ? editableDiv.tagName : null,
                roleTextboxTag: roleTextbox ? roleTextbox.tagName : null,
                hasLocalStorage: (() => { try { return !!window.localStorage; } catch(e) { return false; } })(),
            };
        }""")
        logger.info(f"   textarea: {'OK' if checks.get('textarea') else 'MISSING'}")
        logger.info(f"   contentEditable: {checks.get('contentEditableTag') or 'MISSING'}")
        logger.info(f"   role=textbox: {checks.get('roleTextboxTag') or 'MISSING'}")
        logger.info(f"   localStorage: {'OK' if checks.get('hasLocalStorage') else 'UNAVAILABLE'}")
        if not checks.get("textarea") and not checks.get("contentEditableTag") and not checks.get("roleTextboxTag"):
            errors.append("no input element found — chat UI incomplete")
        if not checks.get("hasLocalStorage"):
            errors.append("localStorage not available")

        # 4. Captcha SDK 检测
        captcha_objects = await page.evaluate("""() => {
            const found = [];
            // 阿里云验证码 SDK 全局对象
            if (typeof window.AWSC !== 'undefined') found.push('AWSC');
            if (typeof window.NVC !== 'undefined') found.push('NVC');
            if (typeof window.AliyunCaptcha !== 'undefined') found.push('AliyunCaptcha');
            if (typeof window.__captcha_init !== 'undefined') found.push('__captcha_init');
            // 检查页面中是否注入了 captcha 相关 script
            const scripts = Array.from(document.querySelectorAll('script[src]'))
                .filter(s => /captcha|aliyun/.test(s.src));
            if (scripts.length > 0) found.push('captcha_script_tags:' + scripts.length);
            return found;
        }""")
        if captcha_objects:
            logger.info(f"   captcha SDK objects: {captcha_objects}")
        else:
            logger.info("  captcha SDK objects: none detected (may load on demand)")

        await page.close()

        if errors:
            logger.warning("CONNECTIVITY CHECK WARNINGS:")
            for e in errors:
                logger.warning(f"  ! {e}")
        else:
            logger.info("connectivity check PASSED")

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            to_remove = [
                uid for uid, e in self._pages.items()
                if now - e["last_used"] > CAPTCHA_PAGE_IDLE_TTL
            ]
            for uid in to_remove:
                await self._close_page(uid)
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} idle pages")

    async def _human_mouse_motion(self, page, viewport_w=1265, viewport_h=1281):
        """Random curved mouse motion to satisfy Aliyun's trajectory check."""
        steps = random.randint(8, 14)
        x = random.randint(100, viewport_w - 100)
        y = random.randint(100, viewport_h - 100)
        await page.mouse.move(x, y)
        for _ in range(steps):
            x = max(50, min(viewport_w - 50, x + random.randint(-200, 200)))
            y = max(50, min(viewport_h - 50, y + random.randint(-200, 200)))
            await page.mouse.move(x, y, steps=random.randint(5, 15))
            await asyncio.sleep(random.uniform(0.03, 0.15))

    async def _close_page(self, user_id: str):
        entry = self._pages.pop(user_id, None)
        if entry:
            try:
                await entry["page"].close()
            except Exception:
                pass

    async def _get_or_create_page(self, user_id: str, jwt_token: str) -> Dict:
        if user_id in self._pages:
            entry = self._pages[user_id]
            entry["last_used"] = time.monotonic()
            return entry

        page = await self._context.new_page()

        async def _permanent_route(route):
            try:
                await route.continue_()
            except Exception:
                pass

        await page.route("**/api/v2/chat/completions**", _permanent_route)

        # 导航到 chat.z.ai，带超时保护
        try:
            resp = await page.goto(
                "https://chat.z.ai",
                wait_until="domcontentloaded",
                timeout=20000,
            )
            if resp is None or resp.status >= 400:
                logger.warning(f"page load failed for {user_id}: status={resp.status if resp else 'N/A'}")
                await page.close()
                raise HTTPException(
                    status_code=503,
                    detail="Failed to load chat.z.ai (network may be restricted)",
                )
        except Exception as e:
            await page.close()
            logger.warning(f"page goto error for {user_id}: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Cannot reach chat.z.ai: {e}",
            )

        # 等待 Svelte 异步渲染 chat UI
        try:
            await page.wait_for_selector(
                'textarea, [contenteditable="true"], [role="textbox"]',
                timeout=15000,
            )
        except Exception:
            title = await page.title()
            logger.warning(f"input element did not appear for {user_id}, title: {title}")
            await page.close()
            raise HTTPException(
                status_code=503,
                detail=f"chat.z.ai chat UI did not render (title: {title})",
            )

        await page.evaluate(
            """(t) => { localStorage.setItem('token', t); }""",
            jwt_token,
        )
        await page.reload(wait_until="domcontentloaded")
        await asyncio.sleep(3)

        entry = {
            "page": page,
            "lock": asyncio.Lock(),
            "last_used": time.monotonic(),
        }
        self._pages[user_id] = entry
        return entry

    async def get_token(self, jwt_token: str) -> str:
        user_id = extract_user_id(jwt_token)
        logger.info(f"token requested for user_id={user_id[:12]}...")
        entry = await self._get_or_create_page(user_id, jwt_token)

        async with entry["lock"]:
            try:
                return await self._trigger_captcha(entry)
            except HTTPException:
                # 触发失败时清理可能的坏页面，下次请求重建
                await self._close_page(user_id)
                raise

    async def _trigger_captcha(self, entry: Dict) -> str:
        page = entry["page"]
        token_holder: Dict[str, Optional[str]] = {"value": None}
        event = asyncio.Event()

        async def capture_route(route):
            try:
                body = route.request.post_data_json  # property, not method
                if body:
                    has_captcha = "captcha_verify_param" in body
                    logger.debug(
                        f"route hit: has_captcha={has_captcha} "
                        f"model={body.get('model', 'N/A')}"
                    )
                    if has_captcha:
                        token_holder["value"] = body["captcha_verify_param"]
                        event.set()
                        await route.abort("blockedbyclient")
                        return
                else:
                    logger.debug("route hit: no JSON body")
            except Exception as e:
                logger.warning(f"route hit: parse error: {e}")
            try:
                await route.continue_()
            except Exception:
                pass

        await page.unroute("**/api/v2/chat/completions**")
        await page.route("**/api/v2/chat/completions**", capture_route)

        # Move mouse along a randomized path before interacting — Aliyun
        # silent-verify scores sessions partially on mouse trajectory, and a
        # cold click with no prior movement reliably fails the check.
        await self._human_mouse_motion(page)

        input_selector = (
            'textarea, [contenteditable="true"], [role="textbox"]'
        )
        try:
            await page.wait_for_selector(input_selector, timeout=5000)
            await page.focus(input_selector)
            await page.locator(input_selector).first.press_sequentially(
                "hi", delay=random.randint(50, 120)
            )
        except Exception:
            logger.warning("input element not found or type failed")
            raise HTTPException(
                status_code=503,
                detail="Chat input not found, page may need re-initialization",
            )

        await asyncio.sleep(random.uniform(0.3, 0.7))

        btn_state = await page.evaluate("""() => {
            const btn = document.querySelector('#send-message-button');
            if (!btn) return 'not_found';
            return btn.disabled ? 'disabled' : 'enabled';
        }""")
        logger.debug(f"send button state after type: {btn_state}")

        if btn_state == 'disabled':
            await page.evaluate("""() => {
                const btn = document.querySelector('#send-message-button');
                if (btn) btn.disabled = false;
            }""")

        # Move mouse to the send button and click via mouse — page.click jumps
        # straight to the element which Aliyun flags as automated.
        btn = await page.query_selector('#send-message-button')
        if not btn:
            raise HTTPException(
                status_code=503,
                detail="Send button not found",
            )
        box = await btn.bounding_box()
        if box:
            target_x = box['x'] + box['width'] / 2
            target_y = box['y'] + box['height'] / 2
            await page.mouse.move(target_x, target_y, steps=20)
            await asyncio.sleep(random.uniform(0.1, 0.3))
            await page.mouse.click(target_x, target_y)
        else:
            await page.click('#send-message-button', timeout=5000)
        logger.debug("clicked send button")

        try:
            await asyncio.wait_for(event.wait(), timeout=CAPTCHA_TOKEN_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="Captcha token generation timed out",
            )

        # Restore permanent pass-through route
        async def _permanent_route(route):
            try:
                await route.continue_()
            except Exception:
                pass

        await page.unroute("**/api/v2/chat/completions**")
        await page.route("**/api/v2/chat/completions**", _permanent_route)

        token = token_holder["value"]
        if not token:
            logger.warning("token capture failed: empty value in intercepted request")
            raise HTTPException(
                status_code=503,
                detail="Failed to capture captcha token",
            )
        logger.info(f"token captured successfully, len={len(token)}")

        # Reset page state so the next request on this same page starts from a
        # clean chat view — leaving the sent "hi" in DOM breaks subsequent
        # captcha triggering.
        try:
            await page.goto("https://chat.z.ai", wait_until="domcontentloaded", timeout=10000)
        except Exception as e:
            logger.debug(f"post-capture reset navigation failed: {e}")

        return token

    async def health(self) -> Dict:
        return {
            "status": "ok" if self._browser and self._browser.is_connected() else "degraded",
            "browser_connected": self._browser.is_connected() if self._browser else False,
            "active_pages": len(self._pages),
        }

    async def shutdown(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        for uid in list(self._pages.keys()):
            await self._close_page(uid)
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser shut down")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

_captcha_browser: Optional[CaptchaBrowser] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _captcha_browser
    _captcha_browser = CaptchaBrowser()
    await _captcha_browser.start()
    yield
    if _captcha_browser:
        await _captcha_browser.shutdown()


app = FastAPI(
    title="Z.ai Captcha Token Service",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


class TokenRequest(BaseModel):
    jwt_token: str


class TokenResponse(BaseModel):
    captcha_verify_param: str
    certify_id: str


@app.post("/token", response_model=TokenResponse)
async def generate_token(req: TokenRequest):
    if not _captcha_browser:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        token = await _captcha_browser.get_token(req.jwt_token)
        try:
            padded = token + "=" * (4 - len(token) % 4)
            data = json.loads(base64.urlsafe_b64decode(padded))
            certify_id = data.get("certifyId", "unknown")
        except Exception:
            certify_id = "unknown"
        return TokenResponse(captcha_verify_param=token, certify_id=certify_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health")
async def health():
    if not _captcha_browser:
        return {"status": "initializing"}
    return await _captcha_browser.health()


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=CAPTCHA_HOST,
        port=CAPTCHA_PORT,
        log_level="info",
        reload=False,
    )
