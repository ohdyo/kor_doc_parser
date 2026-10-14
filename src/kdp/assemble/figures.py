"""그림·차트(기획서 4.3 규칙 11)."""
from __future__ import annotations

import re

_TABLE_IMG = re.compile(r'<img src="(imgs/img_in_[^"]+)" alt="Image"" />')
_OUR_IMG = re.compile(r'<img src="imgs/([^"]+)"')


def fix_table_imgs(html: str, mapping: dict[str, str]) -> str:
    """표 안 그림 경로를 raw/imgs 이름으로 바꾸고 PaddleX 의 따옴표 버그(alt="Image"")를 고친다."""
    def sub(m):
        name = mapping.get(m.group(1))
        src = f"imgs/{name}" if name else m.group(1)
        return f'<img src="{src}" alt="Image" />'

    return _TABLE_IMG.sub(sub, html or "")


def table_img_names(html: str | None) -> list[str]:
    return _OUR_IMG.findall(html or "")
