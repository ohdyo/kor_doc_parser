"""그림 해석 튜닝(D20) 어댑터를 허깅페이스에 올린다 — LoRA 어댑터 + 모델 카드·inference.py(train/figdesc/hf) + 학습 설정·손실 기록·
합성 검증. 토큰은 환경 변수 HF_TOKEN 으로만 받는다(파일·저장소에 적지 않는다). 이 컴퓨터에서 돌린다(서버에 토큰을 두지 않는다).
사용: HF_TOKEN=... python train/figdesc/upload_hf.py runs/figdesc_v1 ohdyo/A.X-kor-image-caption"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="학습 출력 폴더(adapter/·log.jsonl·val.json — train_lora.py 의 out)")
    ap.add_argument("repo", help="허깅페이스 모델 저장소(사용자/이름)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/tune/figdesc_v1.yaml")
    a = ap.parse_args(argv)
    card = (ROOT / "train/figdesc/hf/README.md").read_text(encoding="utf-8")
    if "{RESULTS}" in card or "{TIME}" in card:
        raise SystemExit("모델 카드의 평가·시간 칸을 먼저 채운다")
    files = {"adapter_config.json": a.run / "adapter/adapter_config.json",
             "adapter_model.safetensors": a.run / "adapter/adapter_model.safetensors",
             "README.md": ROOT / "train/figdesc/hf/README.md", "inference.py": ROOT / "train/figdesc/hf/inference.py",
             "train_config.yaml": a.config, "train_log.jsonl": a.run / "log.jsonl", "synth_val.json": a.run / "val.json"}
    with tempfile.TemporaryDirectory() as d:
        for name, src in files.items():
            shutil.copy(src, Path(d) / name)
        api = HfApi(token=os.environ["HF_TOKEN"])
        info = api.upload_folder(folder_path=d, repo_id=a.repo, repo_type="model",
                                 commit_message="그림 해석 튜닝 1차(T1) LoRA 어댑터·모델 카드")
    print("올림", info)


if __name__ == "__main__":
    main()
