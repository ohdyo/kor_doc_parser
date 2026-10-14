"""문서·블록 식별자."""
import hashlib


def doc_id_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def doc_dirname(doc_id: str) -> str:
    """출력 폴더 이름 — ':' 는 SMB 공유에서 못 쓰므로 해시만."""
    return doc_id.split(":", 1)[-1]


def block_id(page_no: int, idx: int) -> str:
    """엔진 결과의 논리 페이지·순번으로 만든다 — 규칙이 바뀌어도 그대로다."""
    return f"p{page_no}.b{idx + 1:02d}"
