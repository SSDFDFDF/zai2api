import re

_THINKING_TAGS = ("details", "think", "reasoning", "thought")
tags_alt = '|'.join(_THINKING_TAGS)

text = 'true">\n> 用户想要搜索北京天气。\n</details>\n<Function_bRiN_Start/>'

cleaned = re.sub(rf'<(?:{tags_alt})[^>]*>.*?</(?:{tags_alt})>\s*', '', text, flags=re.DOTALL)
cleaned = re.sub(rf'^[^<]*?(?:true|false)?"?>\s*(?:>\s*.*?)?</(?:{tags_alt})>\s*', '', cleaned, flags=re.DOTALL)
cleaned = re.sub(rf'<(?:{tags_alt})[^>]*>.*$', '', cleaned, flags=re.DOTALL)

print("IN:", repr(text))
print("OUT:", repr(cleaned))
