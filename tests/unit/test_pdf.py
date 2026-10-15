import json

from kdp.io.pdf import merge_surrogates, open_pdf, read_info, text_chars
from tests.fixtures.synth import nonbmp_pdf


def test_merge_surrogates():
    codes = [0x41, 0xD83D, 0xDE00, 0xD840, 0xDC00, 0xDC00, 0xD83D, 0x42, 0]
    got = [(chr(c), i) for c, i in merge_surrogates(codes)]
    assert got == [("A", 0), ("\U0001F600", 1), ("\U00020000", 3), ("�", 5), ("�", 6), ("B", 7)]


def test_nonbmp_text_layer_is_writable():
    doc = open_pdf(nonbmp_pdf())
    chars = text_chars(doc[0])
    assert [c["c"] for c in chars] == ["\U0001F600", "\U00020000", "C"]
    json.dumps(chars, ensure_ascii=False).encode("utf-8")          # 예전에는 짝 없는 서로게이트로 여기서 실패했다
    assert read_info(doc, "sha256:x", "x.pdf").pages[0].text_chars == 3
