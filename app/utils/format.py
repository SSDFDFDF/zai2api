"""
格式化工具库
"""
from typing import Any

def format_compact_number(value: Any) -> str:
    """格式化大数字，采用 K (千) / M (百万) / B (十亿) 单位展示。"""
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return "0"

    abs_num = abs(number)
    if abs_num >= 1_000_000_000:
        formatted = f"{number / 1_000_000_000:.1f}B"
    elif abs_num >= 1_000_000:
        formatted = f"{number / 1_000_000:.1f}M"
    elif abs_num >= 1_000:
        formatted = f"{number / 1_000:.1f}K"
    else:
        return str(int(number)) if number.is_integer() else f"{number:.1f}"
    
    return formatted.replace(".0B", "B").replace(".0M", "M").replace(".0K", "K")
