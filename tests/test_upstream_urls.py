from app.core.upstream_urls import build_upstream_url, derive_upstream_base_url


def test_derive_upstream_base_url_from_direct_chat_endpoint():
    assert (
        derive_upstream_base_url("https://chat.z.ai/api/v2/chat/completions")
        == "https://chat.z.ai"
    )


def test_derive_upstream_base_url_from_resin_reverse_endpoint():
    endpoint = (
        "http://107.174.40.82:2260/"
        "OzIT6BelAL5aN_o7hKmYsZ1ytDvPUX8E/as/https/chat.z.ai/api/v2/chat/completions"
    )
    assert (
        derive_upstream_base_url(endpoint)
        == "http://107.174.40.82:2260/OzIT6BelAL5aN_o7hKmYsZ1ytDvPUX8E/as/https/chat.z.ai"
    )


def test_build_upstream_url_joins_auxiliary_paths(monkeypatch):
    endpoint = (
        "http://107.174.40.82:2260/"
        "OzIT6BelAL5aN_o7hKmYsZ1ytDvPUX8E/as/https/chat.z.ai/api/v2/chat/completions"
    )
    monkeypatch.setattr("app.core.upstream_urls.settings.API_ENDPOINT", endpoint)
    assert (
        build_upstream_url("/api/v1/auths/")
        == "http://107.174.40.82:2260/OzIT6BelAL5aN_o7hKmYsZ1ytDvPUX8E/as/https/chat.z.ai/api/v1/auths/"
    )
