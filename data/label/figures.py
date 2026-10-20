"""실제 공개 문서 그림의 학습 자료(그림 해석 튜닝 D20, 기획서 7.10) — Claude 라벨러 둘이 따로 정답을 매기고 둘이 같게 본 것만 쓴다.
pack : kdp 출력 폴더(document.json·raw/imgs)마다 설명 대상 그림(describe_targets — 실제 설명과 같은 그림)을 묶는다. 크롭은 모델이 보는
       raw 크롭 그대로, 제목·그림 속 글자·앞뒤 글·물음 목록(kdp.figure_qa — 평가와 같은 물음)과 함께. 담당자·기관장 이름은 ○○○ 로,
       전화번호는 지운다(라벨러에게도 보이지 않게).
merge: 두 라벨러의 정답을 맞춰 학습 자료 JSONL(image·prompt·answer)로 — 둘 다 물음이 맞다고 보고, 보기 물음은 둘 다 같은 보기 하나만
       적었을 때, 단답은 표기가 서로 맞을 때만. 보기 순서는 섞는다.
평가 문서(test2~test5)와 시험 자료 문서(eval/testset/docs.jsonl)는 pack 이 sha256·파일 이름으로 뺀다. 그와 같은 연속 보도(같은 달
기후특성 등)는 이름으로 알 수 없으니 넣지 않게 사람이 고른다(기획서 7.10). 라벨러 지시문은 LABEL_GUIDE.
사용: python data/label/figures.py pack OUT_DIR... --pack PACK_DIR
      python data/label/figures.py merge PACK_DIR labels_A.json labels_B.json --out DATA_DIR/train.jsonl"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kdp.assemble.figures import _caption, _context, describe_targets  # noqa: E402
from kdp.config import DescribeCfg  # noqa: E402
from kdp.enrich.pii import PHONE  # noqa: E402
from kdp.figure_qa import DIAGRAM, NEEDS_X, QUESTIONS, alts, figure_text, nkey, prompt, same_short  # noqa: E402
from kdp.geometry import page_size  # noqa: E402
from kdp.schema import Document  # noqa: E402

EVAL_DOCS = ("test2", "test3", "test4", "test5")    # 평가 문서(eval/judged) — 학습 자료에 넣지 않는다
TESTSET = ROOT / "eval" / "testset" / "docs.jsonl"   # 시험 자료 문서 목록(줄마다 {"name", "sha256", ...}) — 이것도 넣지 않는다
HEADS = (r"(?:기상청장|지청장|질병관리청장|청장|본부장|원장|국장|장관|차관|차장|처장|센터장|소장|단장|위원장)")   # 본문 인용의 기관장 — 부서장
# (과장·실장)은 담당 부서 표가 잡는다. 시장·도지사·군수는 앞에 지명이 와('부산 시장') 넣지 않는다
TITLE = rf"{HEADS}(?=(?:은|는|이|가|의|과|와|도|께서)?(?:[^가-힣]|$))"      # 직함으로 끝나는 말만('장관회의'·'청장년층'·'지원장비' 는 아니다)
NAME_END = r"(?=(?:은|는|이|가|의|과|와|을|를|도|께서|에게)?(?:[^가-힣]|$))"
SURNAMES = set("김이박최정강조윤장임한오서신권황안송전홍유고문양손배백허남심노하곽성차주우구민류나진지엄채원천방공현함변염여추도소석선설"
               "마길연위표명기반라왕금옥육인맹제모탁국어은편용예경봉사부가복태목형피두감호계")
COMPOUND = ("남궁", "황보", "제갈", "선우", "독고", "사공", "서문")
STOP = {"과장", "팀장", "주무관", "사무관", "연구관", "연구사", "담당", "담당자", "책임자", "구분", "부서", "연락처", "전화", "성명",
        "직위", "이름", "직급", "소속", "담당부서", "기관", "비고",
        "올해", "이번", "오늘", "지난해", "당시", "현재", "신임", "전임", "초대", "역대", "또한", "특히", "한편", "이에", "이날", "관련",
        "우리", "정부", "전국", "최근", "지난", "오는", "공동", "주재", "직접", "함께", "모든", "해당", "산하", "부처", "국민", "기자"}
NAME_MAX = 5                  # 문서에 이보다 많이 나오는 낱말은 이름이 아니다(흔한 낱말을 가리면 학습 자료가 망가진다)
LOGO_AREA = 0.03              # 쪽 넓이 대비 — 이보다 작은 그림(로고·아이콘·마크)은 묶지 않는다(설명 대상은 1% 부터)
LABEL_GUIDE = """당신은 한국어 공공 문서의 그래프·지도를 보고 짧은 물음의 정답을 매기는 라벨러입니다. 다른 라벨러도 같은 일을 따로 하고 있으니 남의
결과를 찾거나 참고하지 말고 스스로 판단하세요. 이 정답은 그림 해석 모델의 학습 자료가 됩니다 — 확실한 것만 정답으로 매깁니다.

## 자료
- {pack}/figures.json — 그림마다 image(크롭 PNG), caption(그림 제목), figure_text(그림 속 글자를 기계가 읽은 것 — 틀린 글자가 있을 수
  있음), before/after(그림 앞뒤 본문), questions(qid, qtype, question, options).
- 그림마다 image 를 Read 도구로 직접 열어 보세요. 작은 글씨·막대 위 숫자·범례·축 글자는 파이썬(PIL)으로 잘라 2~3배 키운 사본을
  {zoom}/ 에 만들어 확인하세요.
- 이 JSON 과 그 안의 PNG 말고 다른 파일은 열지 마세요.

## 정답 적는 법
- 보기 물음(options 가 있음): 맞는 보기 하나의 글을 그대로(번호가 아니라 글, 예: "남동쪽"). 둘 이상이 똑같이 맞아 하나로 정할 수
  없으면 ok_question 을 false 로 둡니다.
- 단답 물음(options 가 null): 그림에 적힌 이름 그대로 짧게(예: "2024년", "8월 초순", "60-69세"). 같은 뜻의 다른 표기도 '|' 로 함께
  적되, 맨 앞에는 그림에 적힌 꼴을 둡니다. 막대나 선이 여럿이면 그림 제목이 가리키는 주된 값(보통 막대 또는 첫 번째 계열)으로 봅니다.
- 지도는 위가 북쪽입니다. 지도 땅의 테두리 상자를 3×3 으로 나눈 칸으로 방위를 정하고, '가운데' 는 한가운데 칸입니다. 가장 높은(낮은)
  곳이 넓게 퍼져 있으면 가장 진한(옅은) 곳이 있는 칸으로 답합니다. 편차 지도는 편차 값 자체의 높낮이로, 퍼센타일 지도는 퍼센타일
  값의 높낮이로 봅니다.
- 흐름: 처음에서 끝까지의 큰 흐름. 오르내림만 있고 방향이 없으면 '뚜렷한 흐름 없음'.
- 흐름도 물음(flow_first·flow_last·flow_next): 화살표를 따라가 처음 단계·마지막 단계·'X' 바로 다음 단계의 상자 글을 그대로
  적습니다. 상자 번호(①·1단계·STEP 1)는 떼고, 번호를 붙인 꼴은 '|' 로 함께 적어도 됩니다. 갈림(예/아니오 등)으로 '바로 다음'이
  하나로 정해지지 않거나, 'X' 가 이 그림의 단계 상자 글이 아니거나, 흐름도가 아니면 ok_question 을 false 로.
- 조직도·체계도 물음(org_top·org_parent): 맨 위 상자, 'X' 와 선으로 이어진 바로 위 상자의 글을 그대로. 조직도·체계도가 아니거나
  'X' 가 상자 글이 아니면 false. 상자에 사람 이름이 적혀 있으면 이름은 답·note 어디에도 쓰지 말고 false.
- 로고·엠블럼·아이콘·장식·사진처럼 해석할 내용이 없는 그림이 섞여 있으면 그 그림의 물음은 모두 ok_question 을 false 로 두고
  note 에 '의미 없는 그림' 이라고 적습니다.
- 물음이 이 그림에 맞지 않으면 ok_question 을 false 로, answer 는 빈 글로 둡니다 — 지도가 아닌데 방위를 묻거나 그래프가 아닌데
  흐름을 묻는 경우, 값이 없음, 지도·계열이 여럿이라 어느 것인지 정할 수 없음, 가장 큰 것이 둘 이상 거의 같아 그림으로 가릴 수 없음,
  확대해도 분명하지 않음.
- note: 한 줄 근거. 사람 이름·전화번호는 적지 마세요.

## 출력
모든 물음(qid)을 빠짐없이 매겨 {{"labels": [{{"qid": ..., "ok_question": true, "answer": "...", "note": "..."}}, ...]}} 로 돌려주세요."""


def _contact_words(html: str) -> set[str]:
    """담당 부서 표에서 전화번호가 든 행의 2~4자 낱말(이름 후보)."""
    out = set()
    for row in re.findall(r"<tr\b.*?</tr>", html, re.S | re.I):
        cells = re.sub(r"<[^>]+>", " ", row)
        if PHONE.search(cells):
            out |= set(re.findall(r"(?<![가-힣])[가-힣]{2,4}(?![가-힣])", cells))
    return out


def pii_names(doc: Document) -> set[str]:
    """가릴 이름 — 본문의 기관장 꼴('질병관리청장 ○○○은'·'질병관리청장(○○○)'·'○○○ 기상청장은')과 담당 부서 표의 전화번호 행에서,
    흔한 성으로 시작하는 2~4자 낱말. 표 쪽은 본문·그림 글자에 안 나오는 낱말만(지명·항목 이름을 가리면 학습 물음의 그림 글자가 실제
    추론 때와 달라진다)."""
    blocks = [b for p in doc.pages for b in p.blocks]
    text = "\n".join(b.text or "" for b in blocks)
    names = set(re.findall(rf"{TITLE}\s*\(?\s*([가-힣]{{2,4}}?){NAME_END}", text))
    names |= set(re.findall(rf"(?<![가-힣])([가-힣]{{2,4}})\s+[가-힣]*?{TITLE}", text))
    names = {n for n in names if text.count(n) <= NAME_MAX}
    seen = " ".join([b.text or "" for b in blocks if b.type != "table"] + [b.figure_text or "" for b in blocks])
    for b in blocks:
        if b.type == "table" and ("phone" in (b.pii or []) or PHONE.search(b.html or "")):
            names |= {w for w in _contact_words(b.html or "") if w not in seen}
    return {n for n in names if n not in STOP and (n[:2] in COMPOUND or n[0] in SURNAMES)}


def _phone(m: re.Match) -> str:
    """전화번호 자리 — 번호 앞 괄호까지 잡혔으면 남은 닫는 괄호와 짝이 맞게."""
    g = m.group()
    return "(전화번호 지움" if g.startswith("(") and ")" not in g else "(전화번호 지움)"


def masker(names: set[str]):
    pats = [re.compile(rf"(?<![가-힣]){re.escape(n)}") for n in sorted(names, key=len, reverse=True)]   # 낱말 앞머리에서만

    def clean(s: str) -> str:
        s = PHONE.sub(_phone, s or "")
        for p in pats:
            s = p.sub("○○○", s)
        return s
    return clean


def excluded_docs() -> tuple[set[str], set[str]]:
    """학습 자료에 넣지 않을 문서 — (sha256 들, 파일 이름들). kdp 출력 폴더 이름은 sha256 이라 이름만으로는 못 가린다."""
    shas, names = set(), {f"{n}.pdf" for n in EVAL_DOCS}
    for n in EVAL_DOCS:
        if (ROOT / f"{n}.pdf").exists():
            shas.add(hashlib.sha256((ROOT / f"{n}.pdf").read_bytes()).hexdigest())
    if TESTSET.exists():
        for line in open(TESTSET, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                shas.add(r["sha256"])
                names.add(r["name"])
    return shas, names


def meaningless(b, w: float, h: float) -> bool:
    """해석할 내용이 없는 그림(로고·엠블럼·아이콘·마크·사진) — 라벨 묶음에 넣지 않는다(사용자 2026-10-20). 검출기가 그래프·지도(chart)로
    본 것은 늘 남기고, 그 밖의 그림은 작거나(쪽 넓이 LOGO_AREA 미만), 그림 속 글자가 없거나(사진·삽화), 한 줄 짧은 글뿐(기관 이름·구호 —
    로고·배너)이면 뺀다. w·h 는 쪽 크기(kdp.geometry.page_size)."""
    if b.type == "chart":
        return False
    x0, y0, x1, y1 = b.bbox
    lines = [s.strip() for s in (b.figure_text or "").splitlines() if s.strip()]
    return (x1 - x0) * (y1 - y0) < LOGO_AREA * w * h or not lines or (len(lines) == 1 and len(lines[0]) <= 15)


def _subject(qid: str, text: str) -> str | None:
    """'X' 물음의 X — 그림에서 읽은 글자 가운데 상자 글 같은 줄(2~20자, 숫자만·가린 줄은 아님)을 qid 로 정해 고른다(두 라벨러가 같은
    물음을 받는다). 상자 글이 아니면 라벨러가 물음이 맞지 않다고 뺀다."""
    lines = [s.strip() for s in text.split(" / ")]
    lines = [s for s in lines if 2 <= len(s) <= 20 and not re.fullmatch(r"[\d.,%\s()~-]+", s) and "○○○" not in s and "전화번호" not in s]
    return random.Random(qid).choice(lines) if lines else None


def _questions(tag: str, b, text: str) -> list[dict]:
    """그림 하나의 물음 — 검출기가 그래프·지도(chart)로 본 블록은 그래프·지도 물음만, 그 밖의 그림(image)은 흐름도·조직도 물음도."""
    out = []
    for qt, (q, o) in QUESTIONS.items():
        if b.type == "chart" and qt in DIAGRAM:
            continue
        qid = f"{tag}:{b.id}:{qt}"
        x = _subject(qid, text) if qt in NEEDS_X else None
        if qt in NEEDS_X and not x:
            continue
        out.append({"qid": qid, "qtype": qt, "question": q.replace("{x}", x or ""), "options": o, **({"x": x} if x else {})})
    return out


def pack(a) -> None:
    out = a.pack
    (out / "imgs").mkdir(parents=True, exist_ok=True)
    items, dropped = [], 0
    cfg = DescribeCfg(enabled=True)
    shas, names = excluded_docs()
    for d in a.outs:
        doc = Document.model_validate_json((d / "document.json").read_text(encoding="utf-8"))
        if doc.doc_id.split(":")[-1] in shas or {Path(x).name for x in doc.source.paths} & names:
            print(f"평가·시험 문서라 뺀다: {d}")
            continue
        clean = masker(pii_names(doc))
        tag = d.name[:12]
        for page, b in describe_targets(doc, cfg):
            if meaningless(b, *page_size(page)):
                dropped += 1
                continue
            img = f"imgs/{tag}_{b.id}.png"
            shutil.copyfile(d / "raw" / b.img_path, out / img)
            before, after = _context(doc, b)
            cap = clean(_caption(page, b))
            text = clean(figure_text((b.figure_text or "").splitlines()))
            items.append({"doc": tag, "block": b.id, "image": str((out / img).resolve()), "caption": cap or "(없음)",
                          "figure_text": text, "before": clean(before), "after": clean(after), "questions": _questions(tag, b, text)})
    (out / "figures.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"그림 {len(items)}개(로고·사진 같은 의미 없는 그림 {dropped}개 뺌) · 물음 {sum(len(x['questions']) for x in items)}개"
          f" → {out / 'figures.json'}")


def agreed(qtype: str, la: dict, lb: dict) -> str | None:
    """둘이 같게 본 정답 — 둘 다 물음이 맞다고 봤을 때만. 보기 물음은 둘 다 보기 하나만 적었고 그것이 같을 때 그 보기(정해진 보기 글로 —
    '남동쪽.' 처럼 적어도), 정답이 여럿이면 뺀다(맞는 다른 보기를 틀렸다고 가르치게 된다). 단답은 A 의 표기 가운데 B 와 맞는 첫 것."""
    if not (la and lb and la["ok_question"] and lb["ok_question"]):
        return None
    opts = QUESTIONS[qtype][1]
    if opts:
        ka, kb = alts(la["answer"]), alts(lb["answer"])
        return next((o for o in opts if {nkey(o)} == ka == kb), None)
    sb = alts(lb["answer"])
    return next((x for x in (s.strip() for s in la["answer"].split("|")) if x and same_short(x, sb)), None)


def merge(a) -> None:
    items = json.loads((a.pack / "figures.json").read_text(encoding="utf-8"))
    la = {x["qid"]: x for x in json.loads(Path(a.labels_a).read_text(encoding="utf-8"))["labels"]}
    lb = {x["qid"]: x for x in json.loads(Path(a.labels_b).read_text(encoding="utf-8"))["labels"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = kept = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            for q in it["questions"]:
                n += 1
                ans = agreed(q["qtype"], la.get(q["qid"]), lb.get(q["qid"]))
                if ans is None:
                    continue
                rng = random.Random(q["qid"])
                opts = q["options"]
                if opts:
                    order = rng.sample(opts, len(opts))
                    row = {"prompt": prompt(q["qtype"], it["caption"], it["figure_text"], order), "answer": str(order.index(ans) + 1)}
                else:
                    row = {"prompt": prompt(q["qtype"], it["caption"], it["figure_text"], x=q.get("x")), "answer": ans}
                img = a.pack / "imgs" / Path(it["image"]).name           # pack 폴더를 옮겨도 그 폴더의 그림으로
                if not img.exists():
                    sys.exit(f"그림이 없다: {img}")
                image = os.path.relpath(img.resolve(), a.out.parent.resolve())
                f.write(json.dumps({"image": image, "qtype": q["qtype"], **row, "meta": {"qid": q["qid"]}}, ensure_ascii=False) + "\n")
                kept += 1
    print(f"물음 {n}개 가운데 두 라벨러가 같게 본 {kept}개 → {a.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("outs", nargs="+", type=Path, help="kdp 출력의 문서 폴더(document.json 이 든 곳)")
    p.add_argument("--pack", type=Path, required=True)
    p = sub.add_parser("merge")
    p.add_argument("pack", type=Path)
    p.add_argument("labels_a")
    p.add_argument("labels_b")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    (pack if a.cmd == "pack" else merge)(a)


if __name__ == "__main__":
    main()
