from app.core.headers import (
    CAPTCHA_PROVIDER_SEC_CH_UA,
    CAPTCHA_PROVIDER_USER_AGENT,
    build_dynamic_headers,
)


def test_dynamic_headers_match_captcha_provider_fingerprint():
    headers = build_dynamic_headers("1.2.3")

    assert headers["User-Agent"] == CAPTCHA_PROVIDER_USER_AGENT
    assert headers["sec-ch-ua"] == CAPTCHA_PROVIDER_SEC_CH_UA
    assert headers["sec-ch-ua-mobile"] == "?0"
    assert headers["sec-ch-ua-platform"] == '"Windows"'
    assert headers["Accept-Language"] == "zh-CN,zh;q=0.9,en;q=0.8"
    assert headers["X-FE-Version"] == "1.2.3"
