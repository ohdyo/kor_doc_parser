"""그림 갈래·그림 설명(D19) — 갈래 규칙, 설명할 그림 고르기, 물음·열쇠, 근거 검사, 응답 붙이기."""
import pytest

from kdp.assemble.figures import (attach_descriptions, desc_key, desc_prompt, describe_targets, ground,
                                  route_figures)
from kdp.config import DescribeCfg
from kdp.schema import Block, Document, Page, ParserInfo, Source

ON = DescribeCfg(enabled=True, model="m")
SENTENCES = ("해외유입 말라리아는 주로 아프리카와 동남아시아에서 감염되므로\n"
             "이 지역을 방문하는 여행객은 예방수칙을 준수하여야 합니다.")


def _doc(*blocks, keep=True):
    page = Page(page_no=1, sheet_no=1, size=(595, 842), keep=keep, blocks=list(blocks))
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=1),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=[page])


def _fig(n, typ="chart", text=None, bbox=(70, 100, 520, 400), **kw):
    return Block(id=f"p1.b{n:02d}", type=typ, bbox=bbox, figure_text=text, img_path=f"imgs/p1.b{n:02d}.png", **kw)


@pytest.mark.parametrize("typ, text, want", [
    ("chart", "충북 평균: 20.6℃\n북쪽 지역은 기온이 낮고 남쪽 지역은 높은 편이다", ("image", "chart")),   # 그래프·지도는 글이 길어도
    ("image", None, ("image", "no_text")),
    ("image", "→ ··· ↓", ("image", "no_text")),                   # 기호뿐
    ("image", "45°", ("image", "labels")),                          # 사진 위 각도 표시
    ("image", "머리\n주둥이\n촉수", ("image", "labels")),            # 사진 위 이름표
    ("image", "접수\n검토\n승인\n통보", ("image", "labels")),        # 흐름도 칸 이름
    ("image", SENTENCES, ("text", "sentences")),                    # 문장 위주 글 그림 — 글만으로 뜻이 선다
    ("image", "신청서 접수 및 서류 검토\n현장 실사 및 적격성 평가\n심의위원회 상정 및 의결\n결과 통보 및 사후 관리",
     ("image", "labels")),                                          # 칸 이름이 긴 흐름도 — 끝맺음이 없으면 그림
    ("image", "기온 20.6 21.2 22.1 23.1 18.4 19.3 20.3 강수량 11.2 27.4 19 15 12 6 평년 대비 편차 1.0 1.3 1.6 0.8 0.5 0.3 "
              "퍼센타일 분포 지역별 기상가뭄 발생일수", ("image", "labels")),   # 한글이 길어도 숫자 낱말이 많으면 그림
])
def test_route(typ, text, want):
    b = route_figures(_doc(_fig(1, typ, text))).pages[0].blocks[0]
    assert (b.route, b.route_reason) == want


def test_route_only_kept_figures():
    """버린 그림·병합 조각에는 갈래를 달지 않고, 표 갈래(D18)는 그대로 둔다."""
    doc = route_figures(_doc(_fig(1, "image", SENTENCES, keep=False, drop_reason="logo"),
                             _fig(2, "image", "45°", merged_into="p1.b01"),
                             Block(id="p1.b03", type="table", bbox=(70, 420, 520, 600), html="<table></table>",
                                   route="image", route_reason="failed")))
    assert [(b.route, b.route_reason) for b in doc.pages[0].blocks] == [(None, None), (None, None), ("image", "failed")]


def test_describe_targets():
    """그래프·지도는 제목·글자가 없어도, 그 밖의 그림은 제목이나 그림 속 글자가 있을 때만. 작은 그림(아이콘)·글 갈래·버린 그림·
    크롭 없는 그림·버린 쪽은 뺀다."""
    caption = Block(id="p1.b09", type="figure_title", bbox=(70, 405, 520, 418), text="[ 얼룩날개모기 ]")
    blocks = [_fig(1, "chart"),                                                   # 제목·글자 없는 그래프
              _fig(2, "image", caption_id="p1.b09"),                              # 제목 있는 사진
              _fig(3, "image"),                                                   # 제목·글자 없는 그림(만화 조각)
              _fig(4, "image", "머리 주둥이", bbox=(70, 100, 130, 160)),           # 60×60 — 쪽의 1% 아래
              _fig(5, "image", SENTENCES),                                        # 글 갈래
              _fig(6, "image", "45°", keep=False),
              Block(id="p1.b07", type="image", bbox=(70, 100, 520, 400), figure_text="45°"),   # 크롭 없음
              _fig(8, "image", "45°", bbox=(70, 100, 145, 168)),                  # 75×68 — 1% 를 넘는다
              caption]
    got = [b.id for _, b in describe_targets(route_figures(_doc(*blocks)), ON)]
    assert got == ["p1.b01", "p1.b02", "p1.b08"]
    assert describe_targets(route_figures(_doc(_fig(1, "chart"), keep=False)), ON) == []


def test_prompt_and_key():
    caption = Block(id="p1.b02", type="text", bbox=(70, 405, 520, 418), text="【그림 1】  2026년 9월\n충북 평균기온 분포도")
    doc = _doc(_fig(1, "chart", "기온(℃)\n\n 23.1 \n18.4", caption_id="p1.b02"), caption)
    page, b = doc.pages[0], doc.pages[0].blocks[0]
    p = desc_prompt(doc, page, b)
    assert "[그림] 제목: 【그림 1】 2026년 9월 충북 평균기온 분포도\n" in p and "읽은 글자(틀린 글자가 있을 수 있습니다): 기온(℃) / 23.1 / 18.4\n" in p
    bare = _doc(_fig(1, "chart"))
    assert "[그림] 제목: (없음)\n" in desc_prompt(bare, bare.pages[0], bare.pages[0].blocks[0])
    k = desc_key("m", b, p, 512)
    assert k == desc_key("m", b, p, 512) and k.startswith("sha256:")
    assert len({k, desc_key("n", b, p, 512), desc_key("m", b, p + " ", 512), desc_key("m", b, p, 1024),
                desc_key("m", b.model_copy(update={"id": "p1.b05"}), p, 512)}) == 5     # 생성 한도를 올리면 다시 부른다


REF = "충북 평균: 20.6℃\n기온(℃)\n21.2\n23.1\n18.4\n09-01\n1,250\n【그림 1】 2026년 9월 충북 평균기온 분포도"


def test_ground_keeps_only_grounded_sentences():
    text = ("- **분포도**로, 남서쪽이 23.1℃로 가장 높고 북동쪽이 18.4℃로 가장 낮다. 충북 평균은 20.6℃이다. "
            "남서부의 최고값은 24.5℃이다.\n2. 9월 1일 자료이며 합계는 1250이다. 청주시가 가장 덥다. 남쪽은 高溫 지역이다.")
    kept, dropped, meta = ground(text, REF)
    assert meta == [] and kept == ["분포도로, 남서쪽이 23.1℃로 가장 높고 북동쪽이 18.4℃로 가장 낮다.", "충북 평균은 20.6℃이다.",
                    "9월 1일 자료이며 합계는 1250이다."]      # '09-01' 은 9월 1일, '1,250' 은 1250
    assert dropped == ["남서부의 최고값은 24.5℃이다.", "청주시가 가장 덥다.", "남쪽은 高溫 지역이다."]   # 없는 값·시군구 이름·한자


def test_ground_number_forms():
    """같은 수의 다른 꼴 — 소수 끝 0·앞 0·쉼표."""
    assert ground("평년보다 1℃ 높고 0.50 이다. 9월 1일 값은 2,026 이다.", "편차 +1.0℃\n0.5\n09-01\n2026") == \
        (["평년보다 1℃ 높고 0.50 이다.", "9월 1일 값은 2,026 이다."], [], [])


def test_ground_truncated_drops_last_sentence():
    assert ground("충북 평균은 20.6℃이다. 남서쪽이 23.1", REF, truncated=True) == (["충북 평균은 20.6℃이다."], ["남서쪽이 23.1"], [])
    assert ground("", REF) == ([], [], [])


def test_ground_time_positions_checked_for_sense_not_presence():
    """해·월·일·주는 축의 자리라 그림 글자에 없어도 되지만(OCR 이 축 글자를 자주 놓친다) 말이 되어야 한다 — 달 1~12, 해는 근거
    해 범위(앞뒤 1년, "'23" 은 2023). 기간('지난 3년')은 값이라 근거에 있어야 한다."""
    ref = "2026년(1.1.-6.13.)\n2025년(1.1.-12.31.)\n평년('23-'25) 매개모기"
    ok = ["2026년 자료는 6월 13일까지만 있다.", "2025년은 7월(27주)에 가장 많다.", "평년(2023~2025년)은 7월 말에 가장 높다."]
    assert ground(" ".join(ok), ref) == (ok, [], [])
    bad = ["13월에 가장 많다.", "2031년에 가장 높다.", "지난 3년은 따뜻했다.", "35주차에 가장 많은 45명이다."]
    assert ground(" ".join(bad), ref) == ([], bad, [])    # 없는 달, 범위 밖 해, 근거 없는 기간, 근거 없는 값(45명)
    assert ground("지난 3년은 따뜻했다.", "(b) 지난 3년")[0] == ["지난 3년은 따뜻했다."]
    assert ground("2024년 기온이 가장 높다.", "")[0] == ["2024년 기온이 가장 높다."]   # 근거에 해가 없으면 1900~2100


def test_ground_place_names():
    """시·군·구 이름 꼴은 낱말 머리에서, 뒤에 조사·기호가 올 때만 — '행정구역'·'동남아시아'·'광역시' 는 지명으로 보지 않는다."""
    kept = ["충북의 행정구역별 평균기온 분포다.", "동남아시아에서 감염된다.", "대도시와 광역시는 북반구에 있다."]
    assert ground(" ".join(kept), "충북 평균") == (kept, [], [])
    assert ground("강화군(7)이 가장 높고 파주시와 연천군이 뒤를 잇는다. 강화군에서 많다.", "강화군(7)\n파주시(4)")[1] == \
        ["강화군(7)이 가장 높고 파주시와 연천군이 뒤를 잇는다."]


def test_ground_drops_think_repeats_and_caps_sentences():
    text = "<think>먼저 축을 본다.</think>" + " ".join(["남서쪽이 23.1℃로 높다."] * 3 + [
        "충북 평균은 20.6℃이다.", "기온은 서쪽이 높다.", "21.2 선이 있다.", "18.4 선이 있다."])
    kept, dropped, meta = ground(text, REF)
    assert kept == ["남서쪽이 23.1℃로 높다.", "충북 평균은 20.6℃이다.", "기온은 서쪽이 높다.", "21.2 선이 있다."]
    assert dropped == meta == []


def test_ground_tidies_figure_intro_and_layout_sentences():
    """앞뒤 글과 이어지는 본문 문장으로 — 그림을 소개만 하는 문장·생김새 문장은 빼고(meta), 그림을 가리키는 머리말은 뗀다."""
    text = ("이 그래프는 2026년 9월의 일별 충북 평균 기온 변화를 나타냅니다. 지도에서는 서쪽이 23.1℃로 가장 높다. "
            "이 지도는 서쪽이 가장 높음을 보여 준다. 지도의 오른쪽에는 색상 범례가 제공되어 있다. 북쪽을 가리키는 나침반이 있다. "
            "그림자가 짙은 곳은 북동쪽이다. 충북의 평균기온 분포를 나타낸 지도이다.")
    kept, dropped, meta = ground(text, REF)
    assert kept == ["서쪽이 23.1℃로 가장 높다.", "서쪽이 가장 높음을 보여 준다.", "그림자가 짙은 곳은 북동쪽이다."] and dropped == []
    assert meta == ["이 그래프는 2026년 9월의 일별 충북 평균 기온 변화를 나타냅니다.", "지도의 오른쪽에는 색상 범례가 제공되어 있다.",
                    "북쪽을 가리키는 나침반이 있다.", "충북의 평균기온 분포를 나타낸 지도이다."]


def test_context_in_prompt_and_reference():
    """물음에 그림이 놓인 구역과 읽기 순서로 앞뒤 본문(쪽을 넘어서도)을 넣는다 — 표·다른 그림·표와 그림의 제목·버린 블록은
    건너뛰고 앞은 끝에서, 뒤는 처음부터 CONTEXT_CHARS 자까지. 앞뒤 글의 숫자는 설명의 근거가 된다."""
    long = "가" * 400
    p1 = Page(page_no=1, sheet_no=1, size=(595, 842), blocks=[
        Block(id="p1.b01", type="text", bbox=(70, 60, 520, 80), text=long),
        Block(id="p1.b02", type="text", bbox=(70, 90, 520, 110), text="9월 평균기온은 20.6℃로 평년보다 1.0℃ 높았다."),
        Block(id="p1.b03", type="header", bbox=(70, 10, 520, 30), text="머리말", keep=False, drop_reason="header")])
    p2 = Page(page_no=2, sheet_no=2, size=(595, 842), blocks=[
        Block(id="p2.b01", type="table", bbox=(70, 60, 520, 90), html="<table><tr><td>표 칸</td></tr></table>", caption_id="p2.b02"),
        Block(id="p2.b02", type="text", bbox=(70, 40, 520, 55), text="<표 1> 지점별 기온"),
        _fig(3, "chart", "기온(℃)", caption_id="p2.b04", section_path=["□ 기온", "○ 분포"]).model_copy(update={"id": "p2.b03"}),
        Block(id="p2.b04", type="text", bbox=(70, 405, 520, 418), text="【그림 1】 기온 분포도"),
        Block(id="p2.b05", type="text", bbox=(70, 430, 520, 450), text="하순에는 고온이 이어졌다."),
        Block(id="p2.b06", type="text", bbox=(70, 460, 520, 480), text="나" * 400)])
    doc = route_figures(Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=2),
                                 parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                                 pages=[p1, p2]))
    page, b = doc.pages[1], doc.pages[1].blocks[2]
    prompt = desc_prompt(doc, page, b)
    assert "문서 구역: □ 기온 > ○ 분포\n" in prompt and "[그림] 제목: 【그림 1】 기온 분포도\n" in prompt
    before = prompt.split("앞 글: ", 1)[1].split("\n", 1)[0]
    after = prompt.split("뒤 글: ", 1)[1].split("\n", 1)[0]
    assert before.startswith("…") and before.endswith("9월 평균기온은 20.6℃로 평년보다 1.0℃ 높았다.") and len(before) == 301
    assert "머리말" not in prompt and "표 칸" not in prompt and "<표 1>" not in prompt
    assert after.startswith("하순에는 고온이 이어졌다. 나") and after.endswith("…") and len(after) == 301
    ON_ = DescribeCfg(enabled=True, model="m")
    cache = {desc_key("m", b, prompt, ON_.max_tokens): {"text": "서쪽이 평년보다 1.0℃ 높았다.", "finish_reason": "stop"}}
    out = attach_descriptions(doc, cache, ON_).pages[1].blocks[2]
    assert out.figure_desc == "서쪽이 평년보다 1.0℃ 높았다."             # 1.0 은 앞 글에 적힌 값


def _cache(doc, text, finish="stop", model="m"):
    page, b = doc.pages[0], doc.pages[0].blocks[0]
    return {desc_key(model, b, desc_prompt(doc, page, b), ON.max_tokens): {"text": text, "finish_reason": finish}}


def test_attach_descriptions():
    doc = route_figures(_doc(_fig(1, "chart", "기온(℃)\n23.1\n18.4")))
    out = attach_descriptions(doc, _cache(doc, "남서쪽이 23.1℃로 가장 높다. 최고는 30.5℃다."), ON).pages[0].blocks[0]
    assert out.figure_desc == "남서쪽이 23.1℃로 가장 높다."
    assert out.provenance == [{"rule": "figure_desc", "model": "m", "dropped": ["최고는 30.5℃다."]}]

    none = attach_descriptions(doc, _cache(doc, "최고는 30.5℃다."), ON).pages[0].blocks[0]   # 모두 버리면 설명 없음
    assert none.figure_desc is None and none.provenance[0]["dropped"] == ["최고는 30.5℃다."]

    miss = attach_descriptions(doc, _cache(doc, "남서쪽이 높다.", model="other"), ON).pages[0].blocks[0]   # 다른 모델의 응답
    assert miss.figure_desc is None and miss.provenance == [{"rule": "figure_desc_missing"}]

    assert attach_descriptions(doc, _cache(doc, "남서쪽이 높다."), DescribeCfg()) is doc        # 꺼져 있으면 그대로


def test_attach_uses_figure_inner_text_blocks_and_caption_as_reference():
    """근거는 그림 글자·그림 제목·그림 상자 안에 든 글 블록(글자층이 있어 본문으로 둔 축 이름 등)."""
    doc = route_figures(_doc(_fig(1, "chart", "23.1", caption_id="p1.b03"),
                             Block(id="p1.b02", type="text", bbox=(250, 380, 340, 395), text="주차 44"),
                             Block(id="p1.b03", type="text", bbox=(70, 405, 520, 418), text="[ 2025년 주별 발생 ]")))
    out = attach_descriptions(doc, _cache(doc, "2025년은 44주에 가장 많고 23.1 이다."), ON).pages[0].blocks[0]
    assert out.figure_desc == "2025년은 44주에 가장 많고 23.1 이다."
