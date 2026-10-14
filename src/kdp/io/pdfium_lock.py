"""PDFium 은 스레드 안전하지 않다 — pypdfium2 호출은 이 잠금 안에서만 한다."""
import threading

PDFIUM_LOCK = threading.RLock()
