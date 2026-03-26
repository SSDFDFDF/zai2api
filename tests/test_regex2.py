import re

_THINKING_TAGS = ("details", "think", "reasoning", "thought")
tags_alt = '|'.join(_THINKING_TAGS)

text = '决定调用工具\n<details type="reasoning" done="true">\n> 正在思考'

m_open = re.search(rf'<(?:{tags_alt})[^>]*>', text)

if m_open:
    unclosed = not bool(re.search(rf'</(?:{tags_alt})>', text[m_open.end():]))
    print("Has unclosed:", unclosed)
else:
    print("No open tag")
