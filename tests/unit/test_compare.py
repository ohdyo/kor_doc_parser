"""비교군 시험 지표(eval/compare.py) — 표 격자, 조각 찾기, 지표 계산."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("kdp_compare", Path(__file__).parents[2] / "eval" / "compare.py")
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)


def _box(x):
    return [x, 0, x + 10, 10]


def test_html_grid_spans():
    html = '<table><tr><td rowspan="2">a</td><td colspan="2">b</td></tr><tr><td>c</td><td>d</td></tr></table>'
    assert cmp.html_grid(html) == [2, 3]
    assert cmp.html_grid("<p>x</p>") is None
    assert cmp.th_rows("<table><tr><th>a</th><th>b</th></tr><tr><th>c</th><td>d</td></tr></table>") == 1


def test_present_fuzzy():
    text = cmp.norm("가나다라마바사아자차카타파하 끝")
    assert cmp.present(cmp.norm("다라 마바"), text)
    assert not cmp.present("다라X바", text)                    # 10자 미만은 정확히
    assert cmp.present("가나다라마바사아자X카타", text)         # 12자 중 1자 틀림
    assert not cmp.present("가나다라XX사아자차카타", text)      # 2자 틀림


def test_segments_split_at_rule_or_wide_gap():
    cs = [("가", _box(0)), ("나", _box(10)), ("다", _box(25)), ("라", _box(35))]   # 5pt 틈은 같은 조각
    assert [s for s, _ in cmp._segments(cs)] == ["가나다라"]
    assert [s for s, _ in cmp._segments(cs, [(22, 0, 10)])] == ["가나", "다라"]      # 세로 테두리선
    far = [("가", _box(0)), ("나", _box(10)), ("다", _box(50))]                       # 글자 높이 2배보다 넓은 틈
    assert [s for s, _ in cmp._segments(far)] == ["가나", "다"]


def test_norm_drops_markup_bullets_and_leaders():
    assert cmp.norm("- 가 나 ...... 3") == "가나3"
    assert cmp.norm("• 가 | **나**") == "가나"
    assert cmp.norm("‘가’ “나”") == cmp.norm("'가' \"나\"")


def test_evaluate_counts():
    golden = {"pages": {1: "normal", 2: "toc"}, "junk": [{"page": 1, "kind": "쪽번호", "bbox": [90, 190, 110, 200]}],
              "tables": [{"page": 1, "bbox": [0, 100, 100, 150], "rows": 2, "cols": 3, "header_rows": 1,
                          "caption": [0, 85, 100, 95]}], "figures": [{"page": 1, "bbox": [200, 0, 300, 100]}]}
    segs = {1: [("본문첫째줄", [0, 0, 100, 10]), ("12", [90, 190, 110, 200])],
            2: [("목차", [0, 0, 20, 10]), ("둘째항목", [0, 20, 40, 30])]}
    html = "<table><tr><th>a</th><th>b</th><th>c</th></tr><tr><td>d</td><td>가\\n나</td><td>f</td></tr></table>"
    run = {"seconds": 1.0, "pages": {"1": [
        cmp._blk("text", [0, 0, 100, 10], True, "본문 첫째 줄 $m^2$ 단가 $5$ C:\\nas", "text"),
        cmp._blk("table", [0, 100, 100, 150], True, None, "table", html=html),
        cmp._blk("text", [90, 190, 110, 200], False, "12", "number"),
        cmp._blk("text", [210, 10, 290, 50], True, "그림 속 글자", "text"),
        {**cmp._blk("text", [200, 0, 300, 100], True, "그림을 감싼 문단", "text"), "joined": True}],
        "2": [cmp._blk("text", [0, 0, 20, 10], True, "목차", "paragraph_title"),
              cmp._blk("text", [0, 20, 40, 30], False, "둘째 항목", "content")]}}
    run["pages"]["1"][1]["caption_bbox"] = [10, 86, 90, 94]
    r = cmp.evaluate(segs, golden, run)
    assert r["text"]["normal"] == [1, 1]
    assert r["leak"]["toc"] == [2, 6]                           # 목차 쪽 6자 중 제목 2자가 본문에 남음
    assert r["junk_removed"] == {"쪽번호": [1, 1]}
    assert r["tables"] == {"detected": 1, "grid_ok": 1, "header_ok": 1, "caption_ok": 1, "total": 1}
    assert r["junk_text"] == 2                                   # 표 칸에 글자 그대로 남은 \n, LaTeX 한 개(금액·경로는 아님)
    assert r["figure_chars"] == 5                                # 그림 감싼 문단을 이은 블록은 세지 않는다
