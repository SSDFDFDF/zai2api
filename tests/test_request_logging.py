from app.utils.request_logging import (
    extract_claude_usage,
    extract_openai_usage,
)


def test_extract_openai_usage_supports_cached_prompt_details():
    """Server-reported cached_tokens in prompt_tokens_details should be trusted."""
    usage = extract_openai_usage(
        {
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 45,
                "total_tokens": 165,
                "prompt_tokens_details": {
                    "cached_tokens": 32,
                },
            }
        }
    )

    assert usage == {
        "input_tokens": 120,
        "output_tokens": 45,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 32,
        "total_tokens": 165,
    }


def test_extract_openai_usage_small_input_with_cache_hit():
    """Server-reported cache hits must not be discarded regardless of input size."""
    usage = extract_openai_usage(
        {
            "usage": {
                "prompt_tokens": 741,
                "completion_tokens": 750,
                "total_tokens": 1491,
                "prompt_tokens_details": {
                    "cached_tokens": 512,
                },
            }
        }
    )

    assert usage["cache_read_tokens"] == 512
    assert usage["input_tokens"] == 741
    # No server-reported creation; input < CACHE_CREATION_FLOOR so creation = 0
    assert usage["cache_creation_tokens"] == 0


def test_extract_openai_usage_large_input_with_cache_hit():
    """Partial cache hit with large input."""
    usage = extract_openai_usage(
        {
            "usage": {
                "prompt_tokens": 2364,
                "completion_tokens": 925,
                "total_tokens": 3289,
                "prompt_tokens_details": {
                    "cached_tokens": 1472,
                },
            }
        }
    )

    assert usage["cache_read_tokens"] == 1472
    # estimated_cacheable = (2364 // 128) * 128 = 2304
    # cache_creation = 2304 - 1472 = 832
    assert usage["cache_creation_tokens"] == 832


def test_extract_openai_usage_no_cache_info():
    """When server provides no cache data, estimate creation from stride alignment."""
    usage = extract_openai_usage(
        {
            "usage": {
                "prompt_tokens": 553,
                "completion_tokens": 184,
                "total_tokens": 737,
                "prompt_tokens_details": {},
            }
        }
    )

    assert usage["cache_read_tokens"] == 0
    # input < CACHE_CREATION_FLOOR, so no cache creation estimated
    assert usage["cache_creation_tokens"] == 0


def test_extract_openai_usage_no_cache_info_large_input():
    """When input > CACHE_FLOOR but no cache reported, estimate creation from cacheable."""
    usage = extract_openai_usage(
        {
            "usage": {
                "prompt_tokens": 1496,
                "completion_tokens": 861,
                "total_tokens": 2357,
                "prompt_tokens_details": {},
            }
        }
    )

    assert usage["cache_read_tokens"] == 0
    # estimated_cacheable = (1496 // 128) * 128 = 1408
    assert usage["cache_creation_tokens"] == 1408


def test_extract_openai_usage_with_explicit_creation():
    """Server explicitly reports cache_creation_tokens -- trust it directly."""
    usage = extract_openai_usage(
        {
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 500,
                "total_tokens": 2500,
                "cache_creation_input_tokens": 256,
                "cache_read_input_tokens": 1024,
            }
        }
    )

    assert usage["cache_read_tokens"] == 1024
    assert usage["cache_creation_tokens"] == 256


def test_extract_claude_usage_supports_cache_token_fields():
    usage = extract_claude_usage(
        {
            "usage": {
                "input_tokens": 200,
                "output_tokens": 80,
                "cache_creation_input_tokens": 64,
                "cache_read_input_tokens": 48,
            }
        }
    )

    assert usage == {
        "input_tokens": 200,
        "output_tokens": 80,
        "cache_creation_tokens": 64,
        "cache_read_tokens": 48,
        "total_tokens": 392,
    }
