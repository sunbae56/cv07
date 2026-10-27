# 데이터 개선 실험 가이드 (v3)

Oct 26, 2026 · 

이 문서는 [데이터·EDA 담당 실행 가이드](https://claude.ai/artifact/EGWKfsjbnDFccD8dKvJMLb)(v1, baseline)를 모두 끝내 main을 baseline으로 고정한 뒤, **데이터 쪽 개선 실험을 코드부터 직접 만들며** 진행하는 가이드입니다. 각 실험은 “코드 만들기 → 확인 → 설정 파일 → 학습 → 비교 → PR” 순서이고, 새 기능은 모두 설정 값으로 켜도록 만들어 기본값에서는 baseline 결과가 바뀌지 않습니다.

**v3에서 바뀐 점**: [v2](https://claude.ai/artifact/TK8QJtrqp79hK7oqFt7yfH)의 내용(D1\~D9)은 그대로이고, 잘못된 라벨을 찾아 **고치는** D10과 알약 각인을 **OCR로 읽어 활용하는** D11을 더했습니다. D9 다음의 「v3 추가 실험」 장부터 보면 됩니다.

## 시작 전에: 이 문서의 위치와 실험 목록

**왜 데이터 개선이 중요한가**: 대회 지표 mAP는 **클래스별 점수의 평균**입니다. 박스가 3개뿐인 알약도 30개인 알약과 같은 비중으로 점수에 들어갑니다. 또 IoU 0.75\~0.95를 보므로 박스가 조금만 헐거워도 점수가 사라집니다. 그래서 이 문서의 실험은 **드문 클래스 보강**과 **박스를 헐겁게 만들지 않는 증강**에 집중합니다.

| ID | 실험 | 켜는 방법 | 모델 | 노리는 문제 | 우선 |
| --- | --- | --- | --- | --- | --- |
| D1 | 드문 클래스 오버샘플링 | `data.repeat_threshold: 0.1` (YOLO) / `train.balance_power: 0.5` (R-CNN) | 둘 다 | 클래스 불균형 | 1 |
| D2 | 드문 클래스 합성(copy-paste) | `scripts/make_copy_paste.py` → 데이터 v2 | 둘 다 | 박스 수 자체가 적음 | 2 |
| D3 | 박스가 안전한 증강 | `yolo_args` 프리셋 / `augment.rot90·sat_val·blur` | 둘 다 | 데이터 양, 촬영 조건 차이 | 1 |
| D4 | 입력 해상도 1024 | `data.img_size: 1024` | YOLO | 작은 각인 | 2 |
| D5 | 라벨 정제 강화 | 232장 전수 확인 → `manual_drop.csv` → v1 | 둘 다 | 라벨 오류·빠진 박스 | 1 |
| D6 | 외부 데이터(AI Hub) | 새 make\_external.py → 데이터 버전 v3 / v3all | 둘 다 | 학습에 없는 클래스 | 3 |
| D10 | 잘못된 라벨 찾아 고치기 (v3) | 새 find\_label\_issues.py → `manual_fix.csv` → v1 | 둘 다 | 클래스·박스가 틀린 라벨 | 2 |
| D11 | 각인 OCR 활용 (v3) | 새 src/ocr.py, `family: ocr_rerank` | 둘 다 | 모양·색이 같은 알약 구별 | 3 |

우선 1은 첫째 주, 2는 첫 결과를 본 뒤, 3은 시간이 남을 때입니다. 모델 쪽 개선은 [모델 개선 실험 가이드 (v2)](https://claude.ai/artifact/We19SzDJ8Puc3wMFKTonFC)에 있고, D4는 두 문서가 같이 봅니다.

### 시작 조건: v1 완료 = baseline

main에 아래가 모두 있어야 시작합니다. `git pull origin main` 후 확인하세요.

- baseline zip(v4)의 소스: `scripts/check_data.py`, `clean_labels.py`, `make_reports.py`, `evaluate.py`가 보임
- 데이터 v1 실습 결과: `src/eda.py`의 `class_box_size()`, `01_eda` 4장 셀
- 공용 파일: `data/class_map.json`, `data/splits/v0/`
- `uv run pytest -q` → `4 passed`

### 누가 어느 파일을 고치나 (충돌을 막기 위한 순서)

같은 파일을 두 사람이 동시에 고치면 충돌이 납니다. 그래서 데이터 B는 D1 → D2 → D5를 **차례로**, 데이터 A는 D3를 동시에 진행합니다. 모델·웹 담당의 작업과 연결되는 곳도 아래에 적었습니다.

| 순서 | 담당 | 실험 | 고치는 파일 | 먼저 끝나야 하는 것 |
| --- | --- | --- | --- | --- |
| 0 | 모델 A | 버그 수정 (모델 가이드 0장) | `src/models/yolo.py` | — (팀 전체에서 가장 먼저) |
| 1 | 데이터 B | D1 오버샘플링 | `src/dataset.py`, `src/utils.py`, `scripts/export_yolo.py`, `scripts/check_data.py`, `src/models/yolo.py`(2곳), 새 설정 | 순서 0 병합 |
| 1 | 데이터 A | D3 증강 | `src/transforms.py`, 새 설정 2개 | — |
| 2 | 데이터 B | D2 합성 데이터 | `src/dataset.py`, `scripts/clean_labels.py`, 새 `scripts/make_copy_paste.py`, 새 설정 | D1 병합 |
| 3 | 데이터 B | D5 정제 강화 | `scripts/clean_labels.py` | D2 병합, 비교는 Experimentation Lead의 평가 스크립트 확장(모델 가이드 PS) 병합 |
| 4 | 데이터 B (내려받기는 데이터 A와 나눠도 됨) | D6 외부 데이터 (AI Hub) | 새 `scripts/make_external.py`, 새 설정 2개, `.gitignore` 한 줄 | D2 병합 (records\_to\_coco 사용), 팀 결정 |
| 5 | 데이터 B | D10 라벨 고치기 (v3) | 새 `scripts/find_label_issues.py`, `scripts/clean_labels.py` 전체, 새 `manual_fix.csv` | D5 병합 |
| 6 | 데이터 A (+EL 패키지·등록) | D11 각인 OCR (v3) | 새 `src/ocr.py`, 새 `src/models/ocr_rerank.py`, `src/models/__init__.py`, `pyproject.toml`, 새 설정 | D10 병합 |
| — | 데이터 A·모델 A | D4 해상도 | v1 실습에서 만든 `configs/yolo_img1024.yaml` | — (코드 수정 없음) |

### 실험 하나를 진행하는 방법 (매번 같음)

1. `git switch <내이름>` → `git pull origin main` (다른 사람의 수정을 먼저 받음)
2. 각 실험의 \*\*“코드 만들기”\*\*를 ①부터 순서대로. 코드 블록은 그대로 복사하고, “찾기”는 VS Code에서 Ctrl+F로 그 줄을 찾습니다.
3. **코드 확인**: `uv run pytest -q`(4 passed)와 각 실험의 “확인” 명령. 따옴표·들여쓰기 실수는 여기서 걸립니다.
4. **코드 PR 먼저**: 코드만 커밋해 PR을 올리고 병합합니다(다른 사람이 기다리지 않게). 커밋 때 pre-commit이 import 순서나 빈 줄을 고쳤다고 하면 `git add` → `git commit`을 한 번 더.
5. 학습·비교는 병합 후 여유 있게 돌리고, 점수는 실험 기록표와 실험 이슈에 남깁니다.

아래 코드는 v1을 끝낸 baseline에 이 순서대로 적용해 합성 데이터로 실행까지 확인한 것입니다. 막히면 Experimentation Lead가 가진 **정답본 zip**(모든 단계 적용본)의 같은 파일과 VS Code에서 나란히 비교합니다.

**같은 규칙**: 한 실험에 하나만 바꾸고, baseline과 같은 검증셋(`splits/v0/val.txt`)으로 비교합니다. 모든 명령은 `--config configs/sample_yolo.yaml` 계열로 합성 데이터에서 먼저 연습할 수 있습니다.

## D1. 클래스 불균형: 드문 클래스 이미지 더 자주 학습하기

**한 줄 요약**: 드문 알약이 든 이미지를 한 epoch에 여러 번 보여 줍니다. 데이터를 새로 만들지 않고 설정 값 하나로 켭니다. 증강이 매번 다르게 걸리므로 같은 사진이라도 조금씩 다르게 학습됩니다.

### 어떻게 계산하나 (Repeat Factor Sampling)

클래스 c가 학습 이미지의 f 비율에 나오면 반복 배수는 아래와 같고, 이미지의 배수는 그 안에 든 클래스 중 가장 큰 값입니다.

```latex
r_c = \max\left(1,\ \sqrt{\frac{t}{f_c}}\right)
```

예: `t = 0.1`일 때 학습 이미지 185장 중 2장에만 나오는 알약(f = 0.011)은 약 3배, 10% 이상 나오는 알약은 1배(그대로)입니다. `t`를 키울수록 더 많은 클래스가 반복됩니다.

### 코드 만들기 (데이터 B, 파일 5개 + 설정 1개)

**① `src/dataset.py` — import 2줄 추가** (파일 위쪽, `from collections import defaultdict` 바로 위)

```python
import math
import random
from collections import defaultdict
```

**② `src/dataset.py` — 파일 맨 끝에 붙이기** (이미지별 반복 배수 계산과 반복 목록 만들기)

```python
# ── 5. 클래스 불균형: 드문 클래스 이미지 반복 (실험 D1) ─────────


def repeat_factors(records: list[Record], threshold: float) -> list[float]:
    """이미지별 반복 배수 (LVIS의 Repeat Factor Sampling).

    클래스 c가 학습 이미지의 f_c 비율에 나오면 r_c = max(1, sqrt(threshold / f_c)).
    이미지의 배수는 그 이미지에 든 클래스 중 가장 큰 r_c.
    threshold 0.1이면 이미지의 10%보다 드물게 나오는 클래스부터 반복된다.
    """
    n = max(1, len(records))
    image_count: dict[int, int] = {}
    for r in records:
        for c in set(r["category_ids"]):
            image_count[c] = image_count.get(c, 0) + 1
    rc = {c: max(1.0, math.sqrt(threshold / (k / n))) for c, k in image_count.items()}
    return [max((rc[c] for c in r["category_ids"]), default=1.0) for r in records]


def oversample(records: list[Record], threshold: float, seed: int = 42) -> list[Record]:
    """repeat_factors대로 Record를 반복한 목록. 소수 부분은 확률로 정해 seed가 같으면 결과도 같다."""
    if threshold <= 0:
        return list(records)
    rng = random.Random(seed)
    out: list[Record] = []
    for r, f in zip(records, repeat_factors(records, threshold), strict=True):
        k = int(f) + (1 if rng.random() < f - int(f) else 0)
        out.extend([r] * k)
    return out
```

**③ `src/utils.py` — `# ── JSON ─` 줄 바로 위에 함수 추가** (오버샘플링을 켜면 별도 data.yaml을 써서 baseline의 data.yaml을 덮어쓰지 않게)

```python
def yolo_data_yaml(cfg: dict) -> Path:
    """YOLO 학습에 쓸 data.yaml 경로. 오버샘플링(data.repeat_threshold)을 켜면 별도 파일을 쓴다.

    같은 데이터 버전을 쓰는 baseline 실험의 data.yaml을 덮어쓰지 않기 위해서다.
    """
    d = cfg["data"]
    root = resolve_path(d["yolo_dir"]) / d["version"]
    thr = float(d.get("repeat_threshold", 0) or 0)
    return root / ("data.yaml" if thr <= 0 else f"data_rfs{thr:g}.yaml")
```

**④ `scripts/export_yolo.py` — 4곳 수정**

```python
# (가) import 두 곳
from src.dataset import load_records, load_split, oversample, repeat_factors   # ← 뒤에 두 이름 추가
    xyxy_to_yolo,
    yolo_data_yaml,          # ← from src.utils import ( … ) 목록 맨 아래에 추가
)

# (나) for split 줄 위에 한 줄, 아래에 두 줄
    train_records = []
    for split in ("train", "val"):
        split_records = load_split(records, f"{d['split_dir']}/{d['version']}/{split}.txt")
        if split == "train":
            train_records = split_records
```

(다) `main()` 맨 끝의 `print(f"저장: {root / 'data.yaml'}")` 줄 **바로 아래**에, 같은 들여쓰기(공백 4칸)로 붙입니다.

```python
    # 실험 D1: 드문 클래스가 든 이미지를 목록에 여러 번 적어 더 자주 학습 (baseline data.yaml은 그대로)
    thr = float(d.get("repeat_threshold", 0) or 0)
    if thr > 0:
        repeated = oversample(train_records, thr, seed=cfg.get("seed", 42))
        list_file = root / f"train_rfs{thr:g}.txt"
        # abspath: 링크를 따라가지 않아야 ultralytics가 labels/ 폴더를 찾는다
        list_file.write_text(
            "\n".join(os.path.abspath(root / "images" / "train" / r["file_name"]) for r in repeated)
            + "\n",
            encoding="utf-8",
        )
        rf = repeat_factors(train_records, thr)
        out_yaml = yolo_data_yaml(cfg)
        save_yaml(
            {
                "path": str(root.resolve()),
                "train": list_file.name,
                "val": "images/val",
                "names": dict(enumerate(label_names(class_map))),
            },
            out_yaml,
        )
        print(
            f"오버샘플링(threshold={thr:g}): 학습 이미지 {len(train_records)}장 → {len(repeated)}장 "
            f"(반복된 이미지 {sum(f > 1 for f in rf)}장, 최대 {max(rf, default=1):.1f}배)"
        )
        print(f"저장: {out_yaml}")
```

**⑤ `src/models/yolo.py` — 2곳** (모델 A의 파일이므로 PR에서 모델 A에게 리뷰 요청)

```python
# (가) import: resolve_path → yolo_data_yaml
# 바꾸기 전
from src.utils import get_device, image_id_from_name, resolve_path
# 바꾼 후
from src.utils import get_device, image_id_from_name, yolo_data_yaml

# (나) fit() 안 data_yaml 세 줄 → 한 줄
# 바꾸기 전
        data_yaml = (
            resolve_path(self.cfg["data"]["yolo_dir"]) / self.cfg["data"]["version"] / "data.yaml"
        )
# 바꾼 후
        data_yaml = yolo_data_yaml(self.cfg)  # repeat_threshold를 켜면 data_rfs*.yaml
```

**⑥ `scripts/check_data.py` — 2곳** (점검 도구도 새 파일을 보게)

```python
# (가) import
from src.utils import load_config, resolve_path, yolo_data_yaml
# (나) [3] 부분의 yolo = … 줄
    yolo = yolo_data_yaml(cfg)
```

**⑦ 새 파일 `configs/yolo_rfs.yaml`**

```yaml
# 실험 D1 (데이터): 드문 클래스가 든 이미지를 학습 목록에 더 자주 넣는다 (Repeat Factor Sampling)
# 먼저: uv run python scripts/export_yolo.py --config configs/yolo_rfs.yaml  (data_rfs0.1.yaml 생성)
base: yolo_baseline.yaml
exp_name: 1026_dataB_yolov8s_v0_rfs01
owner: dataB
note: 이미지의 10%보다 드물게 나오는 클래스를 sqrt 배수로 반복 (D1)
data:
  repeat_threshold: 0.1
```

**⑧ 코드 확인** (학습 전에 1분)

```bash
uv run pytest -q                                   # 4 passed (기존 기능이 그대로인지)
uv run python -c "from src.dataset import oversample, load_records; r,_ = load_records('data/raw/train_annotations'); print(len(r), '->', len(oversample(r, 0.1)))"
# → 232 -> (232보다 큰 수). 합성 데이터로 연습 중이면 경로를 data/sample/train_annotations 로
```

오류 없이 숫자가 늘어나면 코드 완성입니다. 여기서 **코드 PR을 먼저 올립니다**(모델 A 리뷰). 다음은 실험 실행입니다.

### 실험 실행 — YOLO

코드 PR이 병합된 뒤, ⑦에서 만든 `configs/yolo_rfs.yaml`로 실험합니다.

```bash
# ① 반복 목록 만들기 (기존 data.yaml은 그대로 두고 data_rfs0.1.yaml을 따로 만듦)
uv run python scripts/export_yolo.py --config configs/yolo_rfs.yaml
# ② 1 epoch으로 동작 확인 → 본 학습
uv run python scripts/run_train.py --config configs/yolo_rfs.yaml --epochs 1 --exp-name test_rfs
uv run python scripts/run_train.py --config configs/yolo_rfs.yaml
```

①의 마지막에 이런 줄이 나오면 켜진 것입니다. (합성 데이터에서 `0.5`로 확인한 출력)

```text
오버샘플링(threshold=0.5): 학습 이미지 64장 → 97장 (반복된 이미지 64장, 최대 1.8배)
저장: …/data/sample/yolo/v0/data_rfs0.5.yaml
```

학습 로그의 `train: Scanning … 97 images, 0 backgrounds`처럼 이미지 수가 늘어 있으면 반영된 것입니다. `uv run python scripts/check_data.py --config configs/yolo_rfs.yaml`의 \[3\]에 `data_rfs0.1.yaml`이 `[OK]`로 보여야 합니다.

### Faster R-CNN: `train.balance_power`

R-CNN은 학습 루프에서 드문 클래스 이미지를 더 자주 뽑는 샘플러가 이미 들어 있습니다. `configs/rcnn_balance.yaml`은 `balance_power: 0.5`만 켭니다(0 끔, 1이면 빈도의 역수만큼). 이 설정 파일은 모델 B가 모델 가이드 T2에서 만듭니다(코드 수정 없음). 내보내기 없이 바로 학습합니다.

```bash
uv run python scripts/run_train.py --config configs/rcnn_balance.yaml
```

### 같이 알아 둘 것

- 모델 담당의 **T2 `cls_pw`**(드문 클래스의 손실을 키움)는 같은 문제를 모델 쪽에서 푸는 방법입니다. **둘을 한 실험에 동시에 켜지 말고** 각각 baseline과 비교한 뒤, 둘 다 효과가 있을 때만 합칩니다.
- 너무 세게 반복하면 드문 알약 몇 장을 외우는 과적합이 납니다. 검증 점수가 떨어지면 `t`를 0.05로 낮춰 봅니다.
- 검증셋은 반복하지 않습니다. 점수 비교 기준은 그대로입니다.

## D2. 드문 클래스 합성 데이터 v2 (copy-paste)

**한 줄 요약**: 학습셋에서 드문 알약을 박스 모양대로 잘라, 알약이 4개 미만인 다른 학습 이미지의 빈 곳에 붙여 새 이미지를 만듭니다. D1이 같은 사진을 반복한다면, D2는 **드문 알약이 다른 배경·위치·방향으로 나오는 새 사진**을 만듭니다. 잘라낸 사각형이 곧 새 박스라서 박스가 정확합니다.

### 코드 만들기 (데이터 B, D1 병합 후)

**① COCO 저장 함수를 공용 위치로 옮기기**: 합성 데이터도 정제와 같은 COCO 형식으로 저장하므로, `scripts/clean_labels.py`에 있던 함수를 `src/dataset.py`로 옮겨 두 스크립트가 같이 씁니다.

1. `scripts/clean_labels.py`에서 `def to_coco(` 함수 전체(다음 `def main` 바로 앞까지)를 **잘라내기(Ctrl+X)**
2. `src/dataset.py`의 `# ── 5. 클래스 불균형` 줄(D1에서 만듦) **바로 위에 붙여넣기**
3. 붙여넣은 함수의 첫 줄을 아래처럼 바꾸기 (이름 + 설명 한 줄)

```python
def records_to_coco(records: list[Record], categories: dict[int, str]) -> dict:
    """Record 목록을 COCO 형식 dict로 (정제·합성 데이터 저장용). bbox는 COCO xywh."""
```

**② `scripts/clean_labels.py` — import 한 줄 추가** (옮긴 함수를 예전 이름으로 불러옴)

```python
from src.dataset import load_records
from src.dataset import records_to_coco as to_coco   # ← 추가
```

**③ 새 파일 `scripts/make_copy_paste.py`** (전체를 그대로 복사)

```python
"""드문 클래스 합성 데이터 만들기: 알약을 잘라 다른 학습 이미지에 붙인다 (실험 D2, 데이터 B).

실행:
    uv run python scripts/make_copy_paste.py --config configs/base.yaml --version v2
    uv run python scripts/make_copy_paste.py --config configs/yolo_v1_clean.yaml --version v2   # 정제 v1 위에

결과:
    data/synth/v2/images/          원본 이미지(하드링크 또는 복사) + 합성 이미지 cp_*.png  — 커밋하지 않음
    data/annotations/v2.json       원본 + 합성 어노테이션 (COCO)                          — 커밋하지 않음
    data/annotations/v2_copy_paste.csv  어떤 알약을 어디에 붙였는지 기록 (파일 이름만)      — 커밋
    data/splits/v2/                train = 원본 train + 합성, val = 원본 val 그대로        — 커밋

규칙
  - 학습셋(train)의 알약만 잘라 학습셋 이미지에만 붙인다. 검증셋 이미지는 절대 쓰지 않는다(점수 부풀림 방지).
  - 박스 수가 --min-boxes 보다 적은 클래스만 채운다. 이미지당 알약은 과제 조건(4개)을 넘지 않는다.
  - 붙일 자리는 기존 박스와 겹치지 않는 곳만 고른다. 잘라낸 사각형 = 새 박스라서 박스가 정확하다.
  - 잘라낸 알약은 90도 단위로 돌리거나 뒤집어 다양하게 붙인다.
설정 파일에서 data 블록의 train_images·train_annotations·version 세 줄을 바꾸면 이 데이터로 학습한다.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import Counter

import cv2
import numpy as np

from src.dataset import load_records, load_split, read_image_rgb, records_to_coco
from src.interfaces import MAX_PILLS_PER_IMAGE, Record
from src.utils import load_config, resolve_path, save_json


def _link_or_copy(src, dst) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)  # 하드링크: 용량을 쓰지 않고, Windows에서도 관리자 권한 없이 된다
    except OSError:
        shutil.copy(src, dst)


def _overlaps(box, boxes, margin: int = 4) -> bool:
    x1, y1, x2, y2 = box
    for bx1, by1, bx2, by2 in boxes:
        if x1 < bx2 + margin and bx1 < x2 + margin and y1 < by2 + margin and by1 < y2 + margin:
            return True
    return False


def _augment_crop(crop: np.ndarray, rng: random.Random) -> np.ndarray:
    crop = np.rot90(crop, rng.choice([0, 1, 2, 3]))
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    return np.ascontiguousarray(crop)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--version", default="v2", help="새 데이터 버전 이름")
    ap.add_argument(
        "--min-boxes", type=int, default=10, help="학습셋 박스가 이보다 적은 클래스를 채움"
    )
    ap.add_argument(
        "--per-image", type=int, default=2, help="합성 이미지 한 장에 붙일 최대 알약 수"
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cfg = load_config(args.config)
    d = cfg["data"]
    records, categories = load_records(d["train_annotations"])
    split_dir = resolve_path(d["split_dir"])
    train = load_split(records, split_dir / d["version"] / "train.txt")
    src_dir = resolve_path(d["train_images"])

    counts = Counter(c for r in train for c in r["category_ids"])
    need = {c: args.min_boxes - n for c, n in counts.items() if n < args.min_boxes}
    crops = {c: [] for c in need}
    for r in train:
        for box, c in zip(r["boxes"], r["category_ids"], strict=True):
            if c in crops:
                crops[c].append((r, box))
    backgrounds = [r for r in train if len(r["boxes"]) < MAX_PILLS_PER_IMAGE]
    print(f"학습셋 클래스 {len(counts)}개 중 박스 {args.min_boxes}개 미만: {len(need)}개")
    if not need or not backgrounds:
        print("채울 클래스가 없거나 붙일 배경 이미지가 없습니다.")
        return

    # 실제 데이터는 data/, 합성 연습 데이터는 data/sample/ 아래에 만든다
    data_root = resolve_path(d["class_map"]).parent
    out_dir = data_root / "synth" / args.version / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in records:  # 원본 전체(검증 포함)를 같은 폴더에 연결해 train_images 하나로 읽게 한다
        _link_or_copy(src_dir / r["file_name"], out_dir / r["file_name"])

    cache: dict[str, np.ndarray] = {}

    def image(rec: Record) -> np.ndarray:
        if rec["file_name"] not in cache:
            cache[rec["file_name"]] = read_image_rgb(src_dir / rec["file_name"])
        return cache[rec["file_name"]]

    synth: list[Record] = []
    next_id = max((r["image_id"] for r in records), default=0) + 1
    log = []
    tries = 0
    while any(v > 0 for v in need.values()) and tries < 5000:
        tries += 1
        bg = rng.choice(backgrounds)
        canvas = image(bg).copy()
        h, w = canvas.shape[:2]
        boxes = [list(b) for b in bg["boxes"]]
        cats = list(bg["category_ids"])
        room = min(args.per_image, MAX_PILLS_PER_IMAGE - len(boxes))
        pasted = 0
        for c in sorted(need, key=lambda k: -need[k]):
            if pasted >= room or need[c] <= 0:
                continue
            src_rec, (x1, y1, x2, y2) = rng.choice(crops[c])
            crop = image(src_rec)[int(y1) : int(y2), int(x1) : int(x2)]
            if crop.size == 0:
                continue
            crop = _augment_crop(crop, rng)
            ch, cw = crop.shape[:2]
            if ch >= h or cw >= w:
                continue
            for _ in range(50):
                px, py = rng.randint(0, w - cw), rng.randint(0, h - ch)
                box = [px, py, px + cw, py + ch]
                if not _overlaps(box, boxes):
                    canvas[py : py + ch, px : px + cw] = crop
                    boxes.append([float(v) for v in box])
                    cats.append(c)
                    need[c] -= 1
                    pasted += 1
                    log.append({"source_image": src_rec["file_name"], "category_id": c})
                    break
        if pasted == 0:
            continue
        name = f"cp_{len(synth) + 1:04d}_{os.path.splitext(bg['file_name'])[0]}.png"
        ok, buf = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        buf.tofile(str(out_dir / name))  # 한글 경로에서도 저장되게
        synth.append(
            Record(
                image_id=next_id + len(synth),  # 원본 id와 겹치지 않는 새 번호
                file_name=name,
                width=w,
                height=h,
                boxes=boxes,
                category_ids=cats,
            )
        )
        for row in log[-pasted:]:
            row.update({"synth_image": name, "background_image": bg["file_name"]})

    ann_dir = data_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    save_json(records_to_coco(records + synth, categories), ann_dir / f"{args.version}.json")
    with open(ann_dir / f"{args.version}_copy_paste.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["synth_image", "background_image", "source_image", "category_id"]
        )
        writer.writeheader()
        writer.writerows(log)

    new_split = split_dir / args.version
    new_split.mkdir(parents=True, exist_ok=True)
    (new_split / "train.txt").write_text(
        "\n".join([r["file_name"] for r in train] + [r["file_name"] for r in synth]) + "\n",
        encoding="utf-8",
    )
    shutil.copy(split_dir / d["version"] / "val.txt", new_split / "val.txt")

    after = Counter(c for r in train + synth for c in r["category_ids"])
    still = [c for c in counts if after[c] < args.min_boxes]
    print(f"합성 이미지 {len(synth)}장, 붙인 알약 {len(log)}개")
    print(f"드문 클래스 최소 박스 수: {min(counts.values())} → {min(after.values())}")
    if still:
        print(
            f"목표를 다 못 채운 클래스 {len(still)}개 (붙일 자리 부족). --per-image를 늘려 보세요."
        )
    print(f"저장: {out_dir}, {ann_dir / (args.version + '.json')}, {new_split}")
    rel = data_root.relative_to(resolve_path(".")).as_posix()
    print("설정 파일 data 블록:")
    print(f"  train_images: {rel}/synth/{args.version}/images")
    print(f"  train_annotations: {rel}/annotations/{args.version}.json")
    print(f"  version: {args.version}")


if __name__ == "__main__":
    main()
```

**④ 새 파일 `configs/yolo_v2_copypaste.yaml`**

```yaml
# 실험 D2 (데이터): 드문 클래스 알약을 잘라 다른 학습 이미지에 붙인 데이터 v2로 학습
# 먼저: uv run python scripts/make_copy_paste.py --config configs/base.yaml --version v2
#       uv run python scripts/export_yolo.py --config configs/yolo_v2_copypaste.yaml
base: yolo_baseline.yaml
exp_name: 1027_dataB_yolov8s_v2_copypaste
owner: dataB
note: 학습셋 박스 10개 미만 클래스를 copy-paste로 보강 (D2)
data:
  train_images: data/synth/v2/images
  train_annotations: data/annotations/v2.json
  version: v2
```

**⑤ 코드 확인**

```bash
uv run pytest -q                                          # 4 passed
uv run python scripts/clean_labels.py --help              # ①·②를 옮긴 뒤에도 오류 없이 도움말
uv run python scripts/make_copy_paste.py --help           # --min-boxes, --per-image 옵션이 보이면 성공
```

`ImportError: cannot import name 'records_to_coco'`가 나오면 ①의 이름 바꾸기를 빠뜨린 것입니다. 확인이 끝나면 코드 PR을 먼저 올리고, 아래 “실행”으로 넘어갑니다.

### 실행

```bash
# ① v2 데이터 만들기 (원본 v0 기준. 정제 v1 위에 만들려면 --config configs/yolo_v1_clean.yaml)
uv run python scripts/make_copy_paste.py --config configs/base.yaml --version v2
# ② YOLO 형식으로 내보내기 → ③ 학습
uv run python scripts/export_yolo.py --config configs/yolo_v2_copypaste.yaml
uv run python scripts/run_train.py   --config configs/yolo_v2_copypaste.yaml
```

합성 데이터에서 `--min-boxes 25`로 확인한 ①의 출력입니다. 마지막 세 줄이 설정 파일에 넣을 값입니다(`yolo_v2_copypaste.yaml`에 이미 적혀 있음).

```text
학습셋 클래스 8개 중 박스 25개 미만: 8개
합성 이미지 36장, 붙인 알약 60개
드문 클래스 최소 박스 수: 10 → 29
설정 파일 data 블록:
  train_images: data/sample/synth/v2/images
  train_annotations: data/sample/annotations/v2.json
  version: v2
```

### 옵션

| 옵션 | 기본 | 뜻 |
| --- | --- | --- |
| `--min-boxes` | 10 | 학습셋 박스가 이보다 적은 클래스를 이 수만큼 채움 |
| `--per-image` | 2 | 합성 이미지 한 장에 붙일 최대 알약 수 (알약은 합쳤 최대 4개) |
| `--version` | v2 | 만들 데이터 버전 이름. 설정을 바꿔 여러 버전(v2, v3)을 비교할 수 있음 |
| `--seed` | 42 | 같은 값이면 팀원 누가 돌려도 같은 v2가 나옴 |

### 반드시 눈으로 확인

`data/synth/v2/images/cp_*.png`를 모두 열어 봅니다. 박스까지 그려 보려면 데이터 baseline 가이드의 “전체 이미지 확인” 코드에서 `configs/base.yaml`을 `configs/yolo_v2_copypaste.yaml`로 바꾸어 실행합니다(결과는 `outputs/eda_review/`). 합성 데이터로 확인했을 때 붙인 알약의 박스는 알약 테두리에 딱 맞았고 기존 알약과 겹치지 않았습니다. 실제 사진에서는 다음을 봅니다.

- 붙인 사각형 가장자리가 배경과 너무 달라 티가 나지 않는지 (촬영 배경이 비슷하면 문제없음)
- 그림자·조명 방향이 어색해 모델이 “붙인 티”를 단서로 배울 만큼인지

티가 심하면 `--per-image 1`로 줄이거나 D1만 쓰고, 그 판단을 실험 기록에 남깁니다. 눈으로 본 합성 이미지는 대회 이미지이므로 PR에 올리지 않고 디스코드로 공유합니다.

### 점수를 속이지 않기 위한 규칙 (스크립트가 지킴)

- **검증셋 이미지는 자르지도, 배경으로 쓰지도 않습니다.** 합성 데이터에서 확인한 결과 검증 이미지 사용 0건, 라벨 이상(범위 밖·5개 이상·겹침) 0건이었습니다.
- `splits/v2/val.txt`는 v0의 검증셋을 그대로 복사합니다. 그래서 baseline과 점수를 바로 비교할 수 있습니다.

### 커밋할 것 / 하지 않을 것

| 파일 | 커밋 |
| --- | --- |
| `data/annotations/v2_copy_paste.csv` (어느 알약을 어디에 붙였는지, 파일 이름만) | 예 |
| `data/splits/v2/` | 예 |
| `data/synth/v2/images/`, `data/annotations/v2.json` | 아니오 (대회 이미지, `.gitignore`로 막힘). 팀원은 같은 명령을 실행하면 같은 파일을 얻음 |

## D3. 박스가 안전한 증강

**한 줄 요약**: 이번 지표는 박스 정확도를 보므로, **박스를 헐겁게 만드는 증강은 피하고** 뒤집기·90도 회전·밝기·채도처럼 박스가 정확히 따라가는 증강을 씁니다.

| 증강 | 박스 | 이번 대회 판단 |
| --- | --- | --- |
| 좌우·상하 뒤집기 | 정확히 따라감 | 안전. 알약은 위아래가 없음 |
| 90도 단위 회전 | 정확히 따라감 | 안전 |
| 임의 각도 회전 (`degrees`) | **커짐**: 기울어진 알약을 감싸는 더 큰 사각형 | 주의. 길쭉한 알약일수록 헐거워져 IoU 0.9 이상에서 손해 가능 |
| 밝기·채도·명도 | 그대로 | 안전 |
| 색상(hue) | 그대로 | 주의. 알약 색이 구분 단서라 0으로 둠 |
| 약한 흐림 | 그대로 | 약하게만. 각인이 사라지면 역효과 |

### YOLO: `configs/yolo_aug_safe.yaml`

YOLO는 증강을 코드가 아니라 `yolo_args`로 정하므로 **새 설정 파일 하나**만 만듭니다(데이터 A). `plots: true`를 쓰므로 모델 A의 버그 수정(모델 가이드 0장)이 main에 병합된 뒤 실행합니다.

**새 파일 `configs/yolo_aug_safe.yaml`**

```yaml
# 실험 D3 (데이터, YOLO): 박스를 헐겁게 만들지 않는 증강 조합
# 임의 각도 회전(degrees)은 박스가 커져서 mAP@[0.75:0.95]에 불리할 수 있어 0으로 둔다.
base: yolo_baseline.yaml
exp_name: 1026_dataA_yolov8s_v0_augsafe
owner: dataA
note: 상하·좌우 뒤집기, 색상(hue) 고정, 채도·명도·크기만 변화 (D3)
yolo_args:
  fliplr: 0.5
  flipud: 0.5
  degrees: 0.0
  hsv_h: 0.0       # 알약 색은 구분 단서라 바꾸지 않음
  hsv_s: 0.4
  hsv_v: 0.3
  scale: 0.3
  translate: 0.1
  mosaic: 1.0
  close_mosaic: 10
  plots: true      # outputs/<실험>/yolo_run/train_batch*.jpg 로 증강 결과 확인 (커밋 금지)
```

```bash
uv run python scripts/run_train.py --config configs/yolo_aug_safe.yaml --epochs 1 --exp-name test_augsafe
```

`outputs/test_augsafe/yolo_run/train_batch0.jpg`를 열면 증강이 적용된 학습 이미지 16장에 박스가 그려져 있습니다. **박스가 알약을 정확히 따라가는지**, 뒤집힌 사진이 섞여 있는지 확인한 뒤 본 학습을 돌립니다. 이 그림은 대회 이미지라 디스코드로만 공유합니다.

> 기존 `configs/yolo_aug_rotate.yaml`은 `degrees: 180`(임의 각도 회전)입니다. `yolo_aug_safe`와 **나란히 비교**해 보면 위 표의 “주의”가 실제로 점수에 어떻게 나타나는지 보고서에 쓸 수 있습니다. 두 설정 모두 이번 수정으로 `plots: true`가 정상 동작합니다(이전 소스는 `plots`를 넣으면 학습이 시작되지 않는 버그가 있었음).

### Faster R-CNN: `configs/rcnn_aug.yaml`

R-CNN은 `src/transforms.py`의 증강을 씁니다. 지금은 뒤집기와 밝기·대비만 있으므로, **90도 회전, 채도·명도, 약한 흐림**을 직접 추가합니다(데이터 A).

**① `src/transforms.py` — `def get_transforms(` 줄 바로 위에 클래스 3개 추가**

```python
class Rotate90:
    """90도 단위 회전 (실험 D3). 박스가 축에 맞춰 그대로 돌아가서 박스가 헐거워지지 않는다.

    임의 각도 회전은 회전된 박스를 감싸는 더 큰 박스가 되어 정답이 헐거워진다.
    mAP@[0.75:0.95]처럼 박스 정확도를 보는 지표에서는 90도 단위가 안전하다.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # 90, 180, 270도 (시계 반대 방향)
            for _ in range(k):
                w = img.shape[1]
                img = np.rot90(img)  # 반시계 90도: (x, y) → (y, w - x)
                b = boxes.copy()
                boxes = np.stack([b[:, 1], w - b[:, 2], b[:, 3], w - b[:, 0]], axis=1).astype(
                    np.float32
                )
        return img, boxes, labels


class SaturationValue:
    """채도·명도만 조금 바꾼다 (실험 D3). 색상(hue)은 알약 구분 단서라 그대로 둔다."""

    def __init__(self, saturation: float = 0.3, value: float = 0.2, p: float = 0.5):
        self.saturation, self.value, self.p = saturation, value, p

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= 1.0 + random.uniform(-self.saturation, self.saturation)
            hsv[..., 2] *= 1.0 + random.uniform(-self.value, self.value)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img, boxes, labels


class GaussianBlur:
    """약한 흐림 (실험 D3). 초점이 덜 맞은 사진에 대비한다. 각인이 사라질 만큼 세게 하지 않는다."""

    def __init__(self, max_kernel: int = 5, p: float = 0.3):
        self.max_kernel, self.p = max_kernel, p

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            k = random.choice([k for k in range(3, self.max_kernel + 1, 2)] or [3])
            img = cv2.GaussianBlur(np.ascontiguousarray(img), (k, k), 0)
        return img, boxes, labels
```

**② `src/transforms.py` — `get_transforms()` 안, brightness 두 줄 바로 아래에 여섯 줄 추가** (설정 키와 연결)

```python
    if cfg_augment.get("brightness", 0) > 0:
        ops.append(BrightnessContrast(cfg_augment["brightness"], cfg_augment.get("contrast", 0.2)))
    if cfg_augment.get("rot90", 0) > 0:                       # ← 여기부터 추가
        ops.append(Rotate90(cfg_augment["rot90"]))
    if cfg_augment.get("sat_val", 0) > 0:
        ops.append(SaturationValue(p=cfg_augment["sat_val"]))
    if cfg_augment.get("blur", 0) > 0:
        ops.append(GaussianBlur(p=cfg_augment["blur"]))
    return Compose(ops) if ops else None
```

**③ 새 파일 `configs/rcnn_aug.yaml`** (값은 각 증강을 적용할 확률, 0이면 꺼짐)

```yaml
# 실험 D3 (데이터, R-CNN): transforms.py 증강 조합 (모두 박스를 정확히 따라감)
base: rcnn_baseline.yaml
exp_name: 1026_dataA_frcnn_v0_aug
owner: dataA
note: 뒤집기 + 90도 회전 + 채도·명도 + 약한 흐림 (D3)
augment:
  name: basic
  hflip: 0.5
  vflip: 0.5
  rot90: 0.5
  brightness: 0.2
  sat_val: 0.5
  blur: 0.2
```

**④ 코드 확인** — 회전 후 박스가 실제 알약 위치와 정확히 같은지 숫자로 확인합니다.

```bash
uv run pytest -q     # 4 passed
uv run python -c "
import random, numpy as np
from src.transforms import Rotate90
img = np.zeros((100, 200, 3), np.uint8); img[10:30, 40:90] = 255
for s in range(3):
    random.seed(s); out, b, _ = Rotate90(1.0)(img, np.array([[40, 10, 90, 30]], np.float32), np.array([0]))
    ys, xs = np.where(out[..., 0] > 0); print(b[0].tolist(), [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
"
```

각 줄의 두 목록(박스 좌표, 실제 흰색 영역)이 같으면 성공입니다. 예: `[110.0, 70.0, 160.0, 90.0] [110, 70, 160, 90]`.

90도 회전은 4방향 모두 박스 좌표가 실제 알약 픽셀 위치와 정확히 일치하는 것을 확인했고, 이 설정으로 합성 데이터 학습도 오류 없이 끝났습니다.

**눈으로 확인**: `02_data_check` 노트북의 “증강 결과 보기” 셀에서 `get_transforms(...)`에 넘기는 dict를 아래처럼 바꿔 실행합니다. 같은 사진이 돌아가고 뒤집히는데 박스가 계속 알약에 붙어 있으면 성공입니다.

```python
get_transforms({"name": "basic", "hflip": 0.5, "vflip": 0.5, "rot90": 0.5, "sat_val": 0.5, "blur": 0.2}, train=True)
```

### 새 증강을 직접 만들 때

`transforms.py`에 클래스를 추가하고 `get_transforms()`에 설정 키를 연결합니다(baseline 가이드 5장 B와 같은 방법). 원칙은 두 가지입니다: 이미지를 움직이면 박스도 같이 움직이고, 기본값은 0(꺼짐)으로 둡니다.

## D4. 입력 해상도 1024

EDA에서 알약이 이미지 면적의 1\~2% 수준으로 작게 나오면 시도합니다. 640으로 줄이면 각인·글자가 뭉개져 비슷한 알약을 헷갈립니다. **코드 수정은 없습니다.** 모델 담당이 v1 따라하기 실습에서 만든 `configs/yolo_img1024.yaml`(`img_size: 1024`, `batch_size: 8`)을 그대로 씁니다. 학습 시간이 2\~3배 늘고 GPU 메모리를 더 씁니다(부족하면 batch 4). 데이터 담당은 EDA 근거(클래스별 박스 크기 표)를 넘기고, 학습은 GPU가 있는 사람이 돌립니다.

```bash
uv run python scripts/run_train.py --config configs/yolo_img1024.yaml
```

## D5. 라벨 정제 강화

baseline 가이드의 자동 규칙(범위 밖 자르기, 작은 박스 빼기)에 더해, **232장을 사람이 전부 보고 고칠 것을 적는 단계**입니다. 두 명이 나눠 보면 30분입니다. 이미지를 한 번에 만드는 코드는 baseline 가이드(또는 EDA 안내)의 “전체 이미지 확인”을 씁니다.

| 발견 | `manual_drop.csv`에 적는 법 | 결과 |
| --- | --- | --- |
| 알약과 다른 이름(번호)이 붙은 박스 | `파일이름.png,그 박스의 category_id` | 그 박스만 빠짐 |
| 박스가 없는 알약이 있음 (라벨 누락) | `파일이름.png,all` | **이미지 전체가 빠짐 (아래 코드 만들기 후)** |
| 박스가 알약보다 눈에 띄게 크거나 작음 | 심하면 `all`, 약하면 기록만 | 헐거운 정답은 모델에게 헐거운 박스를 가르침 |

라벨이 빠진 알약은 모델이 맞게 찾아도 “오답”으로 배우게 만들기 때문에, 박스를 새로 그릴 도구가 없는 이번 프로젝트에서는 이미지를 빼는 것이 안전합니다. 합성 연습 데이터에도 같은 알약이 두 개인 이미지에서 박스가 하나만 있는 경우가 있어(예: `sample_train_0022.png`), 연습용으로 찾아볼 수 있습니다.

### 코드 만들기 (데이터 B, D2 병합 후): `all`로 이미지 전체 빼기

지금 `clean_labels.py`는 `파일,번호`로 **박스 하나**만 뽑니다. 라벨이 빠진 이미지는 박스를 빼서는 고칠 수 없으므로, 번호 자리에 `all`을 쓰면 이미지 전체를 빼도록 두 곳을 고칩니다.

**① `scripts/clean_labels.py` — `load_manual_drop()` 함수 전체를 아래로 바꾸기**

```python
def load_manual_drop(path) -> set[tuple[str, int | None]]:
    """(file_name, category_id) 목록. category_id를 all 로 적으면 그 이미지 전체를 뺀다
    (라벨이 빠진 알약이 있는 이미지처럼 박스 하나로는 고칠 수 없는 경우)."""
    if not path.exists():
        return set()
    out: set[tuple[str, int | None]] = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = (row.get("category_id") or "").strip()
            out.add((row["file_name"].strip(), None if cid.lower() in ("", "all") else int(cid)))
    return out
```

**② `clean()` 함수 안 — `for r in records:` 바로 아래에 다섯 줄 추가**

```python
    for r in records:
        if (r["file_name"], None) in manual_drop:  # 이미지 전체 제외      ← 여기부터
            log.append(
                {"file_name": r["file_name"], "category_id": None, "action": "drop_image_manual"}
            )
            continue                                                          # ← 여기까지 추가
        w, h = r["width"], r["height"]
```

**③ 코드 확인** (합성 데이터로)

```bash
printf 'file_name,category_id\nsample_train_0022.png,all\n' > data/sample/annotations/manual_drop.csv
uv run python scripts/clean_labels.py --config configs/sample_yolo.yaml --version v1
# → 원본 80장 → 정제 79장, 수정 내역: {'drop_image_manual': 1}
```

`data/sample/annotations/manual_drop.csv`는 연습용이라 커밋하지 않습니다(`data/sample/`은 `.gitignore`). 실제 데이터의 `data/annotations/manual_drop.csv`에는 사람이 확인한 항목만 적습니다.

```csv
file_name,category_id
sample_train_0022.png,all
sample_train_0067.png,all
```

```bash
uv run python scripts/clean_labels.py --config configs/base.yaml --version v1
# → "원본 232장 → 정제 N장, 수정 내역: {'drop_image_manual': …, 'clip': …}"
uv run python scripts/export_yolo.py --config configs/yolo_v1_clean.yaml
uv run python scripts/run_train.py   --config configs/yolo_v1_clean.yaml
```

**공정하게 비교하기**: 정제로 검증셋 정답도 바뀌면, baseline은 옛 정답으로, v1 모델은 고친 정답으로 채점되어 비교가 어긋납니다. baseline을 v1 정답으로 다시 채점합니다(재학습 없음, 1\~2분). 이 --val-config 옵션은 Experimentation Lead가 모델 가이드 PS 단계에서 evaluate.py에 추가하므로, 그 PR이 병합된 뒤 씁니다.

```bash
uv run python scripts/evaluate.py --exp 1028_modelA_yolov8s_v0_baseline --val-config configs/yolo_v1_clean.yaml
# → outputs/<baseline>/eval_…_valv1.json — 이 점수와 v1 모델 점수를 비교
```

## D6. 외부 데이터 (AI Hub) 추가

**한 줄 요약**: 테스트에는 학습에 없는 알약이 있고, 학습 232장으로는 드문 알약도 부족합니다. AI Hub 「경구약제 이미지 데이터」의 **조합 사진**(대회와 같은 촬영 형식)을 **학습셋에만** 더해 새 데이터 버전을 만듭니다. 검증셋은 원본 그대로라 이전 실험과 같은 기준으로 비교됩니다.

| 모드 | 버전 | 무엇을 더하나 | class\_map | 언제 |
| --- | --- | --- | --- | --- |
| `--classes known` | v3 | 대회 학습 클래스로만 된 사진 (드문 클래스 보강) | 그대로 | 먼저. 다른 실험과 바로 비교·앙상블 가능 |
| `--classes all` | v3all | 학습에 없던 알약이 든 사진까지 | `data/class_map_v3all.json` 새로 생김 | known 효과 확인 뒤. 클래스 수가 바뀌어 모든 모델 재학습 |

### 시작 전 팀 결정 (Experimentation Lead·PM(X))

- 외부 데이터 사용이 허용되는지 Kaggle 대회 Overview·Rules와 강사 공지로 다시 확인합니다.
- **`TL_2_조합.zip`, `TS_2_조합.zip`은 대회 원본이라 내려받지도, 쓰지도 않습니다.** 스크립트도 폴더 이름에 이 글자가 있으면 멈춥니다.
- AI Hub 데이터도 재배포 금지입니다. 이미지·JSON을 레포·PR·이슈·디스코드에 올리지 않습니다(`data/` 아래에 두면 `.gitignore`가 막음). API 키는 `.env`에만 둡니다.
- `all` 모드는 클래스 번호가 바뀌어 팀 전체 재학습이 필요하므로 날짜를 정해 한 번만 합니다.

### ① AI Hub에서 내려받기 (처음 한 번)

1. [aihub.or.kr](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=576)에 회원가입 → 「경구약제 이미지 데이터」 페이지에서 **다운로드(활용 신청)** → 승인을 기다립니다.
2. **받을 파일**: Training의 경구약제 **조합** 데이터에서 라벨 `TL_n_조합`(각 수 MB)과 이미지 `TS_n_조합`(각 약 3GB). **n=2는 제외**합니다. 처음에는 `TL_1_조합` + `TS_1_조합` 한 쌍으로 시작하고, 효과가 보이면 3\~8번을 늘립니다. 단일 경구약제(알약 하나만 찍은 사진, 파일당 수십 GB)는 촬영 형식이 테스트와 달라 쓰지 않습니다.
3. **방법 A (웹)**: 데이터 페이지의 파일 트리에서 위 두 파일만 체크해 내려받습니다. **방법 B (aihubshell, Linux·macOS·WSL·Colab)**: 사용법은 [AI Hub aihubshell 가이드](https://www.aihub.or.kr/static/pdf/aihubshell_가이드.pdf)를 따릅니다. 먼저 목록 모드로 파일 이름과 filekey를 확인하고, TL\_2·TS\_2의 filekey는 절대 넣지 않습니다.

```bash
./aihubshell -mode l -datasetkey 576                 # 파일 이름·크기·filekey 목록
./aihubshell -mode d -datasetkey 576 -filekey <TL_1 키>,<TS_1 키> -aihubapikey '<내 API 키>'
```

filekey를 빼면 데이터셋 전체(수백 GB)를 받으니 주의합니다. macOS 기본 bash에서는 한글 이름 조각 파일 병합이 실패한다는 보고가 있어, 그 경우 웹으로 받거나 Colab·WSL을 씁니다.

4. **압축 풀어 놓을 곳** (안쪽 하위 폴더 구조는 상관없음, 스크립트가 끝까지 찾음):

```text
data/aihub/labels/TL_1_조합/...   ← JSON
data/aihub/images/TS_1_조합/...   ← PNG
```

5. **눈으로 확인**: 이미지 몇 장을 열어 대회 사진처럼 976×1280, 밝은 회색 배경인지 봅니다. 디스크는 압축본 포함 받은 용량의 2\~3배를 비워 둡니다. `data/aihub`와 레포는 같은 드라이브에 두어야 다음 단계에서 복사 대신 하드링크가 됩니다.

### ② 코드 만들기 (데이터 B, D2 병합 후)

AI Hub JSON은 대회 JSON과 모양이 거의 같지만, `categories`가 모두 `1 "Drug"`라서 알약 번호를 이미지 정보의 `drug_N`(K-코드, 예: `K-033880`) 또는 `dl_idx`에서 만들어야 합니다. 어느 칸이 대회 번호와 맞는지는 스크립트가 **대회 데이터와 약 이름이 같은 알약**으로 자동 확인하고 결과를 출력합니다.

**① 새 파일 `scripts/make_external.py`** (전체를 그대로 복사)

```python
"""외부 데이터(AI Hub 경구약제 이미지)를 학습셋에 더한 새 데이터 버전 만들기 (실험 D6, 데이터 B).

준비 (데이터 가이드 v2의 D6 ①~②):
    data/aihub/labels/TL_1_조합/...   AI Hub 라벨 JSON (압축 푼 폴더, 여러 개 가능)
    data/aihub/images/TS_1_조합/...   AI Hub 이미지 PNG
    ※ TL_2_조합·TS_2_조합은 대회 원본이라 내려받지도, 쓰지도 않는다.

실행:
    # 기존 클래스만 보강 (class_map 그대로, 먼저 이것부터)
    uv run python scripts/make_external.py --config configs/base.yaml --version v3 --classes known
    # 학습에 없던 알약까지 추가 (data/class_map_v3all.json 이 새로 생김 → 모든 모델 재학습)
    uv run python scripts/make_external.py --config configs/base.yaml --version v3all --classes all

결과:
    data/external/v3/images/          원본 + AI Hub 이미지 (하드링크 또는 복사)  — 커밋하지 않음
    data/annotations/v3.json          원본 + AI Hub 어노테이션 (COCO)            — 커밋하지 않음
    data/annotations/v3_external.csv  추가한 AI Hub 이미지 목록 (파일 이름만)     — 커밋
    data/splits/v3/                   train = 원본 train + AI Hub, val = 원본 val — 커밋
    data/class_map_{version}.json     (--classes all 일 때만) 늘어난 클래스 표    — 커밋

규칙
  - AI Hub 이미지는 학습셋에만 넣는다. 검증셋은 원본 그대로라 이전 실험과 점수를 비교할 수 있다.
  - 대회 학습·테스트 이미지와 내용이 같은 사진은 뺀다 (누수 방지).
  - 박스가 이미지 밖에 있거나, bbox가 4개 숫자가 아니거나, 알약이 4개를 넘는 사진은 뺀다.
  - AI Hub JSON의 categories는 모두 1 "Drug"라서, 알약 번호는 images의 drug_N(K-코드) 또는 dl_idx에서 만든다.
    어느 칸을 쓸지는 대회 데이터와 약 이름이 같은 알약으로 자동 확인한다 (--id-field로 직접 지정 가능).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.dataset import build_class_map, load_records, load_split, records_to_coco
from src.interfaces import MAX_PILLS_PER_IMAGE, Record
from src.utils import load_class_map, load_config, load_json, resolve_path, save_json

BANNED = ("TL_2_조합", "TS_2_조합")  # 대회 원본: 사용 금지


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)  # 하드링크: 같은 드라이브면 용량을 쓰지 않는다
    except OSError:
        shutil.copy(src, dst)


def _num(value) -> int | None:
    digits = re.sub(r"\D", "", str(value or ""))
    return int(digits) if digits else None


def _angle(file_name: str) -> str | None:
    """AI Hub 파일 이름 예: K-001900-016548_0_2_0_2_75_000_200.png → '75' (6번째 칸)."""
    parts = Path(file_name).stem.split("_")
    return parts[5] if len(parts) >= 8 and parts[5].isdigit() else None


def _thumb(path: Path) -> np.ndarray | None:
    """64x64 흑백 축소본 (내용 비교용)."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).astype(np.int16)


def _dhash(thumb: np.ndarray) -> int:
    """밝기 차이로 만든 64비트 지문. 거의 같은 사진이면 지문도 거의 같다 (1차 후보 찾기용)."""
    small = cv2.resize(thumb.astype(np.uint8), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _same_photo(a: np.ndarray, b: np.ndarray) -> bool:
    """2차 확인: 축소본 픽셀 차이 평균이 아주 작으면 같은 사진으로 본다."""
    return float(np.abs(a - b).mean()) < 4.0


def read_aihub(label_dirs: list[Path]) -> tuple[list[dict], Counter]:
    """AI Hub 라벨 JSON(이미지 x 알약 하나당 한 개)을 읽어 알약 단위 목록으로."""
    pills, skipped = [], Counter()
    for root in label_dirs:
        if any(b in str(root) for b in BANNED):
            raise SystemExit(f"사용 금지 폴더입니다 (대회 원본): {root}")
        for jf in sorted(root.rglob("*.json")):
            try:
                data = load_json(jf)
                img = data["images"][0]
                ann = data["annotations"][0]
            except Exception:  # 깨진 JSON
                skipped["JSON 읽기 실패"] += 1
                continue
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                skipped["bbox가 4개 숫자가 아님"] += 1
                continue
            pills.append(
                {
                    "file_name": img["file_name"],
                    "width": int(img.get("width", 0)),
                    "height": int(img.get("height", 0)),
                    "bbox": [float(v) for v in bbox],
                    "dl_name": str(img.get("dl_name") or ""),
                    "drug_N": _num(img.get("drug_N")),
                    "dl_idx": _num(img.get("dl_idx")),
                }
            )
    return pills, skipped


def choose_id_rule(pills: list[dict], comp_names: dict[str, int], field: str, offset: int | None):
    """약 이름이 대회 데이터와 같은 알약으로 '어느 칸 + 몇을 더하면 대회 번호가 되는지' 고른다."""
    if field != "auto" and offset is not None:
        return field, offset, None
    fields = ["drug_N", "dl_idx"] if field == "auto" else [field]
    named = {p["dl_name"]: p for p in pills if p["dl_name"] in comp_names}
    best = None
    for f in fields:
        for k in (0, -1, 1) if offset is None else (offset,):
            ok = sum(
                1 for name, p in named.items() if p[f] is not None and p[f] + k == comp_names[name]
            )
            if best is None or ok > best[2]:
                best = (f, k, ok)
    return best[0], best[1], (best[2], len(named))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml", help="출발 데이터 버전의 설정")
    ap.add_argument("--version", default="v3", help="새 데이터 버전 이름")
    ap.add_argument("--labels", nargs="+", default=["data/aihub/labels"], help="AI Hub 라벨 폴더")
    ap.add_argument("--images", nargs="+", default=["data/aihub/images"], help="AI Hub 이미지 폴더")
    ap.add_argument(
        "--classes",
        choices=["known", "all"],
        default="known",
        help="known: 대회 학습 클래스로만 된 사진만 / all: 새 알약도 추가 (class_map이 바뀜)",
    )
    ap.add_argument(
        "--angles", nargs="*", default=["70", "75", "90"], help="쓸 촬영 각도 (빈 값이면 전부)"
    )
    ap.add_argument(
        "--max-per-class", type=int, default=0, help="클래스당 추가 이미지 상한 (0=없음)"
    )
    ap.add_argument("--max-images", type=int, default=0, help="추가 이미지 전체 상한 (0=없음)")
    ap.add_argument("--id-field", choices=["auto", "drug_N", "dl_idx"], default="auto")
    ap.add_argument("--id-offset", type=int, default=None, help="--id-field와 함께 직접 지정")
    ap.add_argument("--no-dedup", action="store_true", help="테스트 이미지와 내용 비교를 건너뜀")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    records, categories = load_records(d["train_annotations"])
    split_dir = resolve_path(d["split_dir"])
    train = load_split(records, split_dir / d["version"] / "train.txt")
    comp_class_map = load_class_map(d["class_map"])
    known = {c["category_id"] for c in comp_class_map}
    comp_names = {c["name"]: c["category_id"] for c in comp_class_map}

    # 1) AI Hub 라벨 읽기 → 알약 번호 규칙 정하기
    pills, skipped = read_aihub([resolve_path(p) for p in args.labels])
    if not pills:
        raise SystemExit(f"AI Hub 라벨 JSON을 찾지 못했습니다: {args.labels}")
    field, offset, check = choose_id_rule(pills, comp_names, args.id_field, args.id_offset)
    print(f"AI Hub 라벨: 알약 {len(pills)}개 읽음")
    print(f"알약 번호 = {field} + ({offset})", end="")
    if check:
        print(f"  [이름이 같은 알약 {check[1]}종 중 {check[0]}종 번호 일치]")
        if check[1] and check[0] < check[1] * 0.9:
            print(
                "  [주의] 번호가 맞지 않는 알약이 많습니다. --id-field / --id-offset을 확인하세요."
            )
    else:
        print("  (직접 지정)")

    # 2) 이미지 단위로 묶기 + 검사
    by_image: dict[str, list[dict]] = defaultdict(list)
    for p in pills:
        by_image[p["file_name"]].append(p)
    image_paths = {}
    for root in args.images:
        root = resolve_path(root)
        if any(b in str(root) for b in BANNED):
            raise SystemExit(f"사용 금지 폴더입니다 (대회 원본): {root}")
        for path in root.rglob("*.png"):
            image_paths.setdefault(path.name, path)
    comp_files = {r["file_name"] for r in records}
    test_dir = resolve_path(d["test_images"])
    test_files = {p.name for p in test_dir.glob("*.png")} if test_dir.exists() else set()

    names = dict(categories)
    candidates = []
    angles = set(args.angles or [])
    for file_name, items in sorted(by_image.items()):
        if file_name in comp_files or file_name in test_files:
            skipped["대회 이미지와 이름이 같음"] += 1
            continue
        if file_name not in image_paths:
            skipped["이미지 파일 없음"] += 1
            continue
        if angles and _angle(file_name) not in angles:
            skipped["촬영 각도 제외"] += 1
            continue
        w, h = items[0]["width"], items[0]["height"]
        boxes, cats, bad = [], [], False
        for p in items:
            x, y, bw, bh = p["bbox"]
            cid = None if p[field] is None else p[field] + offset
            if cid is None or bw <= 1 or bh <= 1 or x < 0 or y < 0 or x + bw > w or y + bh > h:
                bad = True
                break
            box = [round(x, 1), round(y, 1), round(x + bw, 1), round(y + bh, 1)]
            if box not in boxes:
                boxes.append(box)
                cats.append(cid)
                names.setdefault(cid, p["dl_name"] or str(cid))
        if bad:
            skipped["박스 오류(이미지 밖·크기 0·번호 없음)"] += 1
            continue
        if len(boxes) > MAX_PILLS_PER_IMAGE:
            skipped[f"알약 {MAX_PILLS_PER_IMAGE}개 초과"] += 1
            continue
        if args.classes == "known" and not set(cats) <= known:
            skipped["학습에 없는 알약 포함 (--classes known)"] += 1
            continue
        candidates.append((file_name, w, h, boxes, cats))

    # 3) 테스트 이미지와 내용이 같은 사진 빼기 (지문으로 후보 → 축소본 픽셀로 확인)
    if not args.no_dedup and test_files and candidates:
        print(f"테스트 이미지와 내용 비교 중... (후보 {len(candidates)}장, 몇 분 걸릴 수 있음)")
        tests = [t for p in sorted(test_dir.glob("*.png")) if (t := _thumb(p)) is not None]
        test_hashes = [_dhash(t) for t in tests]
        kept = []
        for c in candidates:
            thumb = _thumb(image_paths[c[0]])
            if thumb is not None:
                h = _dhash(thumb)
                if any(
                    (h ^ th).bit_count() <= 6 and _same_photo(thumb, t)
                    for th, t in zip(test_hashes, tests, strict=True)
                ):
                    skipped["테스트 이미지와 같은 사진"] += 1
                    continue
            kept.append(c)
        candidates = kept

    # 4) 상한 적용 (드문 클래스가 든 사진부터)
    base_count = Counter(c for r in train for c in r["category_ids"])
    candidates.sort(key=lambda c: min(base_count.get(cid, 0) for cid in c[4]))
    added_count: Counter = Counter()
    next_id = max((r["image_id"] for r in records), default=0) + 1
    external: list[Record] = []
    for file_name, w, h, boxes, cats in candidates:
        if args.max_images and len(external) >= args.max_images:
            break
        if args.max_per_class and all(added_count[c] >= args.max_per_class for c in cats):
            continue
        added_count.update(set(cats))
        external.append(
            Record(
                image_id=next_id + len(external),  # 원본 id와 겹치지 않는 새 번호
                file_name=file_name,
                width=w,
                height=h,
                boxes=boxes,
                category_ids=cats,
            )
        )
    if not external:
        raise SystemExit(
            "추가할 이미지가 없습니다. 아래 제외 사유를 확인하세요.\n" + str(dict(skipped))
        )

    # 5) 저장: 이미지 폴더, 어노테이션, 분할, (all이면) 새 class_map
    data_root = resolve_path(d["class_map"]).parent
    out_dir = data_root / "external" / args.version / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = resolve_path(d["train_images"])
    for r in records:  # 원본 전체(검증 포함)도 같은 폴더에 연결해 train_images 하나로 읽게 한다
        _link_or_copy(src_dir / r["file_name"], out_dir / r["file_name"])
    for r in external:
        _link_or_copy(image_paths[r["file_name"]], out_dir / r["file_name"])

    ann_dir = data_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    all_records = records + external
    save_json(records_to_coco(all_records, names), ann_dir / f"{args.version}.json")
    with open(ann_dir / f"{args.version}_external.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "n_pills", "category_ids"])
        for r in external:
            writer.writerow(
                [r["file_name"], len(r["boxes"]), " ".join(map(str, r["category_ids"]))]
            )

    new_split = split_dir / args.version
    new_split.mkdir(parents=True, exist_ok=True)
    (new_split / "train.txt").write_text(
        "\n".join([r["file_name"] for r in train] + [r["file_name"] for r in external]) + "\n",
        encoding="utf-8",
    )
    shutil.copy(split_dir / d["version"] / "val.txt", new_split / "val.txt")

    class_map_path = d["class_map"]
    new_classes = sorted({c for r in external for c in r["category_ids"]} - known)
    if args.classes == "all":
        # 기존 클래스 + 새 클래스. build_class_map은 번호순으로 label_idx를 다시 매긴다
        class_map = build_class_map(all_records, names)
        cm_file = data_root / f"class_map_{args.version}.json"
        save_json(class_map, cm_file)
        class_map_path = cm_file.relative_to(resolve_path(".")).as_posix()

    print("\n제외한 AI Hub 이미지:", dict(skipped) or "없음")
    print(
        f"추가한 AI Hub 이미지 {len(external)}장 (알약 {sum(len(r['boxes']) for r in external)}개)"
    )
    print(f"학습셋: {len(train)}장 → {len(train) + len(external)}장, 검증셋은 그대로")
    print(
        f"새 클래스 {len(new_classes)}개"
        + (" (class_map 그대로)" if args.classes == "known" else "")
    )
    rel = data_root.relative_to(resolve_path(".")).as_posix()
    print("설정 파일 data 블록:")
    print(f"  train_images: {rel}/external/{args.version}/images")
    print(f"  train_annotations: {rel}/annotations/{args.version}.json")
    print(f"  class_map: {class_map_path}")
    print(f"  version: {args.version}")


if __name__ == "__main__":
    main()
```

**② `.gitignore` 한 줄 추가**: `!data/class_map.json` 줄 바로 아래에 넣습니다(`all` 모드의 새 클래스 표를 커밋하기 위해).

```text
!data/class_map_*.json
```

**③ 새 설정 파일 두 개**

```yaml
# configs/yolo_v3_aihub.yaml
# 실험 D6 (데이터): AI Hub 외부 데이터로 기존 클래스 보강 (class_map 그대로)
# 먼저: uv run python scripts/make_external.py --config configs/base.yaml --version v3 --classes known
#       uv run python scripts/export_yolo.py --config configs/yolo_v3_aihub.yaml
base: yolo_baseline.yaml
exp_name: 1029_dataB_yolov8s_v3_aihub
owner: dataB
note: AI Hub 조합 데이터(TL_2·TS_2 제외)로 학습셋 보강, 검증셋은 원본 그대로 (D6)
data:
  train_images: data/external/v3/images
  train_annotations: data/annotations/v3.json
  version: v3
```

```yaml
# configs/yolo_v3all_aihub.yaml
# 실험 D6 (데이터): AI Hub 외부 데이터로 학습에 없던 알약까지 추가 (클래스 수가 늘어남)
# 먼저: uv run python scripts/make_external.py --config configs/base.yaml --version v3all --classes all
#       uv run python scripts/export_yolo.py --config configs/yolo_v3all_aihub.yaml
# class_map이 바뀌므로 이 설정으로 학습한 모델끼리만 앙상블할 수 있다.
base: yolo_baseline.yaml
exp_name: 1030_dataB_yolov8s_v3all_aihub
owner: dataB
note: AI Hub 조합 데이터로 새 알약 클래스 추가 (D6)
data:
  train_images: data/external/v3all/images
  train_annotations: data/annotations/v3all.json
  class_map: data/class_map_v3all.json
  version: v3all
```

**④ 코드 확인**: `uv run python scripts/make_external.py --help`가 옵션 목록을 보여 주고, `uv run ruff check .`와 `uv run pytest -q`가 통과하면 코드 PR을 먼저 올립니다(데이터는 각자 PC에).

### ③ 실행 — 먼저 known (v3)

```bash
uv run python scripts/make_external.py --config configs/base.yaml --version v3 --classes known
uv run python scripts/export_yolo.py --config configs/yolo_v3_aihub.yaml
uv run python scripts/check_data.py --config configs/yolo_v3_aihub.yaml          # [3]에 v3 data.yaml [OK]
uv run python scripts/run_train.py --config configs/yolo_v3_aihub.yaml --epochs 1 --exp-name test_d6
uv run python scripts/run_train.py --config configs/yolo_v3_aihub.yaml
uv run python scripts/evaluate.py --exp 1029_dataB_yolov8s_v3_aihub --class-metrics
```

합성 연습 데이터에 AI Hub 형식을 흉내 낸 가짜 폴더를 붙여 확인한 첫 명령의 출력입니다(실제 데이터는 숫자가 훨씬 큼).

```text
AI Hub 라벨: 알약 80개 읽음
알약 번호 = drug_N + (0)  [이름이 같은 알약 8종 중 8종 번호 일치]
테스트 이미지와 내용 비교 중... (후보 26장, 몇 분 걸릴 수 있음)

제외한 AI Hub 이미지: {'학습에 없는 알약 포함 (--classes known)': 4, '촬영 각도 제외': 10, '박스 오류(이미지 밖·크기 0·번호 없음)': 1, '테스트 이미지와 같은 사진': 1}
추가한 AI Hub 이미지 25장 (알약 48개)
학습셋: 64장 → 89장, 검증셋은 그대로
새 클래스 0개 (class_map 그대로)
```

**출력 읽는 법**

- `이름이 같은 알약 X종 중 Y종 번호 일치`: Y가 X와 거의 같아야 합니다. 크게 다르면 `[주의]`가 찍히니 멈추고 Experimentation Lead와 함께 `--id-field drug_N --id-offset 0`처럼 직접 지정해 다시 실행합니다. 번호가 틀리면 다른 약 이름으로 학습되어 점수가 오히려 떨어집니다.
- `제외한 AI Hub 이미지`: 각 사유의 수를 PR 본문에 적습니다. `테스트 이미지와 같은 사진`이 0이 아니면 누수가 막힌 것입니다.
- 학습 시간은 이미지 수에 비례해 늘어납니다. 처음에는 `--max-images 2000` 또는 `--angles 75 90`으로 줄여 시작해도 됩니다.

### ④ 실행 — 다음 all (v3all)

```bash
uv run python scripts/make_external.py --config configs/base.yaml --version v3all --classes all
uv run python scripts/export_yolo.py --config configs/yolo_v3all_aihub.yaml
uv run python scripts/run_train.py --config configs/yolo_v3all_aihub.yaml
uv run python scripts/make_submission.py --exp 1030_dataB_yolov8s_v3all_aihub   # 제출은 PM(X)
```

- 마지막에 `새 클래스 N개`와 `data/class_map_v3all.json`이 생깁니다. 각 실험은 자기 class\_map 경로를 `outputs/<실험>/config.yaml`에 기록하므로 제출 파일과 데모는 알아서 맞는 번호를 씁니다.
- **새 클래스의 효과는 검증 점수로 보이지 않습니다**(검증셋에 그 알약이 없음). 검증 점수가 크게 떨어지지 않았는지만 보고, 효과는 Kaggle 제출 점수로 판단합니다.
- 앙상블(P2)은 같은 class\_map을 쓴 모델끼리만 합칩니다. v3all 모델과 v0·v3 모델을 섞으면 번호가 어긋납니다.
- 데모의 약 정보 표(제조사·성분)는 새 알약에 대해 비어 있고 이름만 나옵니다.

### 옵션

| 옵션 | 기본 | 뜻 |
| --- | --- | --- |
| `--labels`, `--images` | `data/aihub/labels`, `data/aihub/images` | AI Hub 폴더 (여러 개 가능) |
| `--classes` | known | known / all (위 표) |
| `--angles` | 70 75 90 | 쓸 촬영 각도. 파일 이름 6번째 칸(예: `_75_`). 빈 값이면 전부 |
| `--max-per-class` | 0 (없음) | 클래스당 추가 이미지 상한. 드문 클래스가 든 사진부터 고름 |
| `--max-images` | 0 (없음) | 추가 이미지 전체 상한 |
| `--id-field`, `--id-offset` | auto | 알약 번호를 만들 칸과 더할 값 직접 지정 |
| `--no-dedup` | 꺼짐 | 테스트 이미지와 내용 비교 건너뛰기 (권장하지 않음) |

### 커밋할 것 / 하지 않을 것

| 커밋 | 커밋하지 않음 |
| --- | --- |
| `scripts/make_external.py`, 새 설정 2개, `.gitignore`, `data/annotations/v3_external.csv`(파일 이름만), `data/splits/v3/`, `data/splits/v3all/`, `data/class_map_v3all.json` | `data/aihub/`, `data/external/`, `data/annotations/v3*.json`, 학습 그림, `.env`의 API 키 |

## 추가 실험 (시간이 더 들어도 해 볼 것): D7·D8·D9

D1\~D6을 끝내고 시간이 남으면 합니다. D7은 **판단을 믿을 수 있게** 하고 최종 제출 점수를 올리는 것이라 우선순위가 가장 높습니다. D7의 모델 학습에는 모델 가이드 「추가 실험」의 X0(`--set` 옵션, Experimentation Lead)이 먼저 병합되어 있어야 합니다.

| ID | 실험 | 담당 | 만드는 것 | 비용 |
| --- | --- | --- | --- | --- |
| D7 | K-fold 교차검증 + 전체 데이터 학습 | 데이터 B (분할), 모델 A·B (학습) | 새 `scripts/make_folds.py` | 학습 5배 |
| D8 | 테스트 이미지 의사 라벨링 | 데이터 B | 새 `scripts/make_pseudo.py` | 학습 1번 + 팀 결정 |
| D9 | MixUp·넓은 크기 변화 증강 | 데이터 A | 새 설정 1개 | 학습 1번 |

### D7. K-fold 교차검증과 전체 데이터 학습

**왜**: 검증셋이 수십 장이라 ±0.01 차이는 우연일 수 있습니다. 전체 이미지를 5조각으로 나눠 5번 학습·검증한 **평균**으로 비교하면 판단이 안정됩니다. 또 최종 제출 모델은 검증셋까지 모두 학습에 쓰면(232장 전부) 보통 조금 더 좋아집니다. 5개 fold 모델은 그대로 앙상블(P2)에도 쓸 수 있습니다.

**① 새 파일 `scripts/make_folds.py`** (전체를 그대로 복사)

```python
"""K-fold 교차검증 분할과 전체 학습용 분할 만들기 (실험 D7, 데이터 B).

검증셋이 수십 장뿐이라 점수가 우연히 흔들린다. 전체 이미지를 K조각으로 나눠
K번 학습·검증한 평균으로 판단하면 더 믿을 수 있고, K개 모델은 그대로 앙상블(P2)에 쓴다.

실행:
    uv run python scripts/make_folds.py --config configs/base.yaml --k 5

결과 (모두 커밋, 파일 이름만 들어 있음):
    data/splits/{version}_f1 ~ _f5/   train.txt, val.txt  (전체 이미지를 5조각, 조각마다 검증 1개)
    data/splits/{version}_full/       train.txt = 전체 이미지, val.txt = 원래 검증셋
                                      (최종 제출용 학습. 검증 점수는 학습에 쓴 사진이라 의미 없음)

분할 기준은 split_data.py와 같다: 이미지에서 가장 드문 클래스로 층화한다.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

from src.dataset import load_records
from src.interfaces import SEED
from src.utils import load_config, resolve_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--k", type=int, default=5, help="조각 수")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    version = d["version"]
    split_dir = resolve_path(d["split_dir"])

    records, _ = load_records(d["train_annotations"])
    names = [r["file_name"] for r in records]
    freq = Counter(c for r in records for c in r["category_ids"])
    keys = [
        min(r["category_ids"], key=lambda c: freq[c]) if r["category_ids"] else -1 for r in records
    ]
    key_count = Counter(keys)
    keys = [k if key_count[k] >= args.k else "rare" for k in keys]  # 조각 수보다 적으면 묶음
    if min(Counter(keys).values()) >= args.k:
        folds = StratifiedKFold(args.k, shuffle=True, random_state=SEED).split(names, keys)
        how = "층화"
    else:
        folds = KFold(args.k, shuffle=True, random_state=SEED).split(names)
        how = "무작위 (클래스당 이미지가 너무 적음)"

    for i, (tr, va) in enumerate(folds, start=1):
        out = split_dir / f"{version}_f{i}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "train.txt").write_text(
            "\n".join(sorted(names[j] for j in tr)) + "\n", encoding="utf-8"
        )
        (out / "val.txt").write_text(
            "\n".join(sorted(names[j] for j in va)) + "\n", encoding="utf-8"
        )
        missing = {c for j in va for c in records[j]["category_ids"]} - {
            c for j in tr for c in records[j]["category_ids"]
        }
        warn = f"  [주의] 검증 조각에만 있는 클래스 {len(missing)}개" if missing else ""
        print(f"{out.name}: train {len(tr)}장 / val {len(va)}장{warn}")

    full = split_dir / f"{version}_full"
    full.mkdir(parents=True, exist_ok=True)
    (full / "train.txt").write_text("\n".join(sorted(names)) + "\n", encoding="utf-8")
    shutil.copy(split_dir / version / "val.txt", full / "val.txt")
    print(f"{full.name}: train {len(names)}장 (전체) / val = 원래 검증셋 (점수 참고용 아님)")
    print(f"분할 방식: {how}")


if __name__ == "__main__":
    main()
```

**② 실행 (데이터 B)**: 분할 파일은 파일 이름만 들어 있어 커밋합니다.

```bash
uv run python scripts/make_folds.py --config configs/base.yaml --k 5
```

합성 연습 데이터(80장)에서 확인한 출력입니다.

```text
v0_f1: train 64장 / val 16장
...
v0_f5: train 64장 / val 16장
v0_full: train 80장 (전체) / val = 원래 검증셋 (점수 참고용 아님)
분할 방식: 층화
```

**③ 학습 (모델 담당)**: 같은 설정 파일에 `--set data.version=v0_f1`처럼 버전만 바꿔 5번 돌립니다. YOLO는 fold마다 내보내기도 해야 합니다.

```bash
# fold 1 (2~5도 숫자만 바꿔 반복)
uv run python scripts/export_yolo.py --config configs/yolo_baseline.yaml --set data.version=v0_f1
uv run python scripts/run_train.py --config configs/yolo_baseline.yaml \
    --exp-name 1103_modelA_yolov8s_cv_f1 --set data.version=v0_f1
```

판단은 5개 fold의 `metrics.json` 점수 평균으로 합니다(모델 가이드 실습 Step 5의 실험 기록표에 5줄이 생깁니다). fold마다 검증 사진이 달라 **baseline 검증 점수와 직접 비교하지 않고**, 비교할 두 설정을 둘 다 5-fold로 돌려 평균끼리 비교합니다. 시간이 5배라 마지막에 남은 후보 2\~3개에만 씁니다.

**④ 최종 제출용 전체 학습**: 모델 가이드 P5에서 `configs/yolo_full.yaml`(`data.version: v0_full`)로 합니다. 이때 나오는 검증 점수는 학습에 쓴 사진이라 의미가 없습니다.

### D8. 테스트 이미지 의사 라벨링 (팀 결정 필요)

**왜**: 테스트 842장은 라벨이 없지만, 모델이 확신하는 예측을 정답처럼 붙여 학습하면 같은 촬영 환경의 사진을 더 볼 수 있습니다. **대회 규칙이 테스트 이미지를 학습에 쓰는 것을 허용하는지 먼저 확인합니다** (금지되면 하지 않음). 모델의 실수도 함께 배울 수 있어 확신도 기준을 높게(0.7 이상) 시작합니다.

**① 새 파일 `scripts/make_pseudo.py`** (전체를 그대로 복사)

```python
"""테스트 이미지에 모델이 확신하는 예측을 정답처럼 붙여 학습셋에 더하기 (실험 D8, 의사 라벨링).

라벨이 없는 테스트 842장 중, 모든 박스를 높은 확신도로 찾은 사진만 골라 학습에 쓴다.
대회 규칙이 테스트 이미지를 학습에 쓰는 것을 허용하는지 먼저 확인한다 (팀 결정).

실행:
    # 1) 가장 좋은 모델(또는 앙상블)로 테스트 예측을 먼저 만든다
    uv run python scripts/make_submission.py --exp <최고 실험>            # → outputs/<실험>/preds_test.json
    # 2) 확신하는 사진만 골라 새 데이터 버전 만들기
    uv run python scripts/make_pseudo.py --config configs/base.yaml --exp <최고 실험> --version v4pseudo

결과:
    data/pseudo/v4pseudo/images/        원본 + 고른 테스트 이미지 (하드링크 또는 복사) — 커밋하지 않음
    data/annotations/v4pseudo.json      원본 + 의사 라벨 (COCO)                       — 커밋하지 않음
    data/annotations/v4pseudo_pseudo.csv 고른 테스트 이미지 목록                       — 커밋
    data/splits/v4pseudo/               train = 원본 train + 고른 테스트, val = 원본 val — 커밋

규칙
  - 검증셋은 그대로 둔다 (의사 라벨이 검증 점수를 속이지 않게).
  - 한 사진의 박스가 모두 --thr 이상이고, --low ~ --thr 사이의 애매한 박스가 없을 때만 쓴다.
  - 의사 라벨을 만든 모델과 같은 데이터로 다시 학습하면 그 모델의 실수도 함께 배운다.
    그래서 확신도 기준은 높게(0.7 이상) 시작한다.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter

import cv2
import numpy as np

from src.dataset import load_records, load_split, records_to_coco
from src.infer import test_pred_file
from src.interfaces import MAX_PILLS_PER_IMAGE, Record
from src.utils import (
    label_to_category,
    load_class_map,
    load_config,
    load_json,
    predictions_from_json,
    resolve_path,
    save_json,
)


def _link_or_copy(src, dst) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml", help="출발 데이터 버전의 설정")
    ap.add_argument("--exp", required=True, help="테스트 예측(preds_test.json)을 만든 실험")
    ap.add_argument("--tta", action="store_true", help="preds_test_tta.json 사용")
    ap.add_argument("--version", default="v4pseudo")
    ap.add_argument("--thr", type=float, default=0.7, help="이 확신도 이상인 박스만 정답으로")
    ap.add_argument("--low", type=float, default=0.1, help="low~thr 사이 박스가 있으면 사진 제외")
    ap.add_argument("--max-images", type=int, default=0, help="추가할 테스트 이미지 상한 (0=없음)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    records, categories = load_records(d["train_annotations"])
    split_dir = resolve_path(d["split_dir"])
    train = load_split(records, split_dir / d["version"] / "train.txt")
    l2c = label_to_category(load_class_map(d["class_map"]))

    pred_file = test_pred_file(args.exp, args.tta)
    if not pred_file.exists():
        raise SystemExit(f"{pred_file} 이 없습니다. 먼저 make_submission.py --exp {args.exp}")
    preds = predictions_from_json(load_json(pred_file))
    test_dir = resolve_path(d["test_images"])

    chosen, skipped = [], Counter()
    for p in sorted(preds, key=lambda x: -float(x["scores"].max()) if len(x["scores"]) else 0):
        scores = p["scores"]
        sure = scores >= args.thr
        unsure = (scores >= args.low) & ~sure
        if int(sure.sum()) == 0:
            skipped["확신하는 박스 없음"] += 1
            continue
        if bool(unsure.any()):
            skipped["애매한 박스 있음"] += 1
            continue
        if int(sure.sum()) > MAX_PILLS_PER_IMAGE:
            skipped["박스가 4개 초과"] += 1
            continue
        chosen.append(p)
        if args.max_images and len(chosen) >= args.max_images:
            break

    next_id = max((r["image_id"] for r in records), default=0) + 1
    pseudo: list[Record] = []
    for p in chosen:
        path = test_dir / p["file_name"]
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            skipped["이미지 읽기 실패"] += 1
            continue
        h, w = img.shape[:2]
        keep = p["scores"] >= args.thr
        boxes = [[round(float(v), 1) for v in b] for b in p["boxes"][keep].tolist()]
        cats = [l2c[int(lbl)] for lbl in p["labels"][keep].tolist()]
        pseudo.append(
            Record(
                image_id=next_id + len(pseudo),
                file_name=p["file_name"],
                width=w,
                height=h,
                boxes=boxes,
                category_ids=cats,
            )
        )
    if not pseudo:
        raise SystemExit(f"고른 사진이 없습니다. --thr를 낮춰 보세요. 제외 사유: {dict(skipped)}")
    if {r["file_name"] for r in pseudo} & {r["file_name"] for r in records}:
        raise SystemExit("테스트 이미지 이름이 학습 이미지와 겹칩니다. 파일 이름을 확인하세요.")

    data_root = resolve_path(d["class_map"]).parent
    out_dir = data_root / "pseudo" / args.version / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = resolve_path(d["train_images"])
    for r in records:
        _link_or_copy(src_dir / r["file_name"], out_dir / r["file_name"])
    for r in pseudo:
        _link_or_copy(test_dir / r["file_name"], out_dir / r["file_name"])

    ann_dir = data_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    save_json(records_to_coco(records + pseudo, categories), ann_dir / f"{args.version}.json")
    with open(ann_dir / f"{args.version}_pseudo.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "n_pills", "category_ids"])
        for r in pseudo:
            writer.writerow(
                [r["file_name"], len(r["boxes"]), " ".join(map(str, r["category_ids"]))]
            )

    new_split = split_dir / args.version
    new_split.mkdir(parents=True, exist_ok=True)
    (new_split / "train.txt").write_text(
        "\n".join([r["file_name"] for r in train] + [r["file_name"] for r in pseudo]) + "\n",
        encoding="utf-8",
    )
    shutil.copy(split_dir / d["version"] / "val.txt", new_split / "val.txt")

    print(
        f"테스트 {len(preds)}장 중 {len(pseudo)}장 사용 (확신도 {args.thr} 이상), 제외: {dict(skipped)}"
    )
    print(f"학습셋: {len(train)}장 → {len(train) + len(pseudo)}장, 검증셋은 그대로")
    rel = data_root.relative_to(resolve_path(".")).as_posix()
    print("설정 파일 data 블록:")
    print(f"  train_images: {rel}/pseudo/{args.version}/images")
    print(f"  train_annotations: {rel}/annotations/{args.version}.json")
    print(f"  version: {args.version}")


if __name__ == "__main__":
    main()
```

**② 설정 파일** `configs/yolo_v4pseudo.yaml`

```yaml
# configs/yolo_v4pseudo.yaml
# 실험 D8 (데이터): 테스트 이미지 의사 라벨을 더한 데이터로 학습
# 먼저: uv run python scripts/make_pseudo.py --config configs/base.yaml --exp <최고 실험> --version v4pseudo
#       uv run python scripts/export_yolo.py --config configs/yolo_v4pseudo.yaml
base: yolo_baseline.yaml
exp_name: 1103_dataB_yolov8s_v4pseudo
owner: dataB
note: 테스트 예측 확신도 0.7 이상 사진 추가 (D8)
data:
  train_images: data/pseudo/v4pseudo/images
  train_annotations: data/annotations/v4pseudo.json
  version: v4pseudo
```

**③ 실행**

```bash
uv run python scripts/make_submission.py --exp <최고 실험>          # outputs/<최고 실험>/preds_test.json 생성
uv run python scripts/make_pseudo.py --config configs/base.yaml --exp <최고 실험> --version v4pseudo
uv run python scripts/export_yolo.py --config configs/yolo_v4pseudo.yaml
uv run python scripts/run_train.py --config configs/yolo_v4pseudo.yaml
```

합성 연습 데이터에서 `--thr 0.5`로 확인한 출력입니다(약한 연습 모델이라 적게 골랐음).

```text
테스트 12장 중 1장 사용 (확신도 0.5 이상), 제외: {'애매한 박스 있음': 4, '확신하는 박스 없음': 7}
학습셋: 64장 → 65장, 검증셋은 그대로
```

- 검증셋은 원본 그대로라 baseline과 바로 비교됩니다.
- 더 좋은 모델(앙상블 제출 파일을 만든 모델)로 만든 의사 라벨일수록 효과가 큽니다. 점수가 오르면 새 모델로 한 번 더 만들어 볼 수 있지만 2회를 넘기지 않습니다.
- 커밋: 스크립트, 설정, `data/annotations/v4pseudo_pseudo.csv`, `data/splits/v4pseudo/`. 커밋하지 않음: `data/pseudo/`, `data/annotations/v4pseudo.json`.

### D9. MixUp·넓은 크기 변화 (YOLO)

D3의 `yolo_aug_safe.yaml` 위에 두 사진을 엷게 겹치는 MixUp과 더 넓은 크기 변화를 더합니다. 박스는 그대로 정확합니다. D3가 baseline보다 좋았을 때만 해 봅니다.

```yaml
# configs/yolo_aug_mixup.yaml
# 실험 D9 (데이터): D3의 안전한 증강 위에 MixUp(두 사진 겹치기)과 더 넓은 크기 변화를 더한다
base: yolo_aug_safe.yaml
exp_name: 1102_dataA_yolov8s_v0_augmixup
owner: dataA
note: augsafe + mixup 0.1 + scale 0.5 (D9)
yolo_args:
  mixup: 0.1
  scale: 0.5
```

```bash
uv run python scripts/run_train.py --config configs/yolo_aug_mixup.yaml --epochs 1 --exp-name test_d9
uv run python scripts/run_train.py --config configs/yolo_aug_mixup.yaml
```

비교 기준은 baseline이 아니라 **D3(`yolo_aug_safe`)** 결과입니다. Ultralytics의 `copy_paste` 옵션은 알약 윤곽(분할 라벨)이 있어야 동작하므로 쓰지 않고, 같은 목적은 D2가 맡습니다.

## v3 추가 실험: 라벨 바로잡기(D10)와 각인 OCR(D11)

v3에서 새로 더한 두 실험입니다. D10은 **잘못된 라벨을 빼지 않고 고치는** 방법, D11은 알약 **각인(식별문자)을 OCR로 읽어** 라벨 점검과 예측 이름 고르기에 쓰는 방법입니다. 둘 다 합성 연습 데이터로 끝까지 돌려 확인했습니다.

| ID | 실험 | 담당 | 만드는 것 | 먼저 끝나야 하는 것 |
| --- | --- | --- | --- | --- |
| D10 | 의심 라벨 자동 찾기 + 고침 목록 | 데이터 B | 새 `scripts/find_label_issues.py`, `scripts/clean_labels.py` 전체 교체, 새 `data/annotations/manual_fix.csv` | D5 병합, 학습된 baseline 실험 하나 |
| D11 | 각인 OCR: 라벨 점검 + 이름 다시 고르기 | 데이터 A (등록·패키지는 Experimentation Lead) | 새 `src/ocr.py`, 새 `src/models/ocr_rerank.py`, `pyproject.toml` 한 줄, `src/models/__init__.py` 등록, 새 설정 | D10 병합, `data/drug_info.json` (v1 웹 실습의 `build_drug_info.py`) |

## D10. 잘못된 라벨 찾아서 고치기

**한 줄 요약**: D5의 `manual_drop.csv`는 잘못된 박스를 **빼기만** 합니다. 232장뿐인 데이터에서는 빼는 것도 손해입니다. D10은 ① 학습된 모델로 **의심 라벨 후보만** 골라 주고, ② 사람이 사진을 보고 확인한 뒤, ③ `manual_fix.csv`에 적으면 **클래스·박스 좌표를 고치거나 빠진 박스를 추가**합니다.

### 찾는 것 (후보 종류)

| 종류 | 뜻 | 기본 처리 제안 |
| --- | --- | --- |
| `wrong_class` | 모델이 확신도 0.5 이상으로 같은 자리(IoU 0.7 이상)에 **다른 알약**을 예측 | `class` (예측 클래스로 바꾸기) |
| `loose_box` | 같은 알약을 찾았지만 박스 겹침이 0.5\~0.75 | `box` (예측 박스로 바꾸기) |
| `missing_label` | 모델이 확신하는데 근처에 정답 박스가 없음 | `add` (예측 박스 추가) |
| `not_found` | 확신도가 낮은 예측까지 봐도 정답 자리에 아무것도 없음 | `check` (사진 보고 판단) |
| `ocr_mismatch` | (D11 `--ocr`) 읽은 각인이 정답 알약보다 다른 알약과 훨씬 비슷함 | `class` |

**모델이 틀린 경우도 섞여 있습니다.** 후보는 “사람이 볼 목록”일 뿐이고, 사진을 보고 맞는 것만 옮깁니다. 검증셋 라벨 오류는 점수 판단 자체를 흔들기 때문에 **검증셋(`--split val`)부터** 확인합니다.

### 코드 만들기 (데이터 B, D5 병합 후)

**① 새 파일 `scripts/find_label_issues.py`** (전체를 그대로 복사)

```python
"""라벨이 잘못됐을 수 있는 박스를 자동으로 찾아 목록으로 만든다 (실험 D10, 데이터 B / D11 --ocr, 데이터 A).

학습된 모델의 예측과 정답을 비교해 사람이 확인할 후보만 골라 준다. 232장을 다 보지 않아도 된다.
후보를 눈으로 확인한 뒤, 맞는 줄만 data/annotations/manual_fix.csv 에 옮기면 clean_labels.py가 고친다.

실행:
    uv run python scripts/find_label_issues.py --exp <학습된 실험> --split val
    uv run python scripts/find_label_issues.py --exp <학습된 실험> --split all --draw      # 확인용 그림까지
    uv run python scripts/find_label_issues.py --exp <학습된 실험> --split all --ocr       # 각인 OCR 비교 추가 (D11)

결과 (대회 데이터에서 나온 것이므로 커밋하지 않음):
    outputs/<실험>/label_issues_<split>.csv   후보 목록 (manual_fix.csv와 같은 앞 네 칸 + 근거)
    outputs/<실험>/label_issues/*.jpg         --draw: 초록=정답, 빨강=예측 (번호는 label_idx)

찾는 것
  wrong_class   모델이 높은 확신도로 같은 자리에 다른 알약을 예측 → 클래스가 틀렸을 수 있음
  loose_box     같은 알약을 찾았지만 박스 겹침(IoU)이 0.5~0.75 → 정답 박스가 헐겁거나 어긋났을 수 있음
  missing_label 모델이 확신하는데 근처에 정답 박스가 없음 → 라벨이 빠졌을 수 있음
  not_found     확신도가 낮은 예측까지 봐도 정답 박스 자리에 아무것도 없음 → 라벨 오류이거나 어려운 사진
  ocr_mismatch  (--ocr) 읽은 각인이 정답 알약보다 다른 알약 각인과 훨씬 비슷함
모델이 틀린 경우도 섞여 있으므로 반드시 사진을 보고 판단한다. 검증셋(val)부터 확인한다.
"""

from __future__ import annotations

import argparse
import csv

import cv2
import torch
from torchvision.ops import box_iou

from src.dataset import load_records, load_split, read_image_rgb
from src.infer import load_experiment
from src.interfaces import OUTPUT_DIR
from src.utils import category_to_label, label_to_category, load_class_map, resolve_path

FIELDS = [
    "file_name",
    "category_id",
    "action",
    "new_value",
    "issue",
    "pred_category_id",
    "score",
    "iou",
    "gt_box",
    "pred_box",
    "ocr_text",
]


def _fmt(box) -> str:
    return " ".join(f"{float(v):.1f}" for v in box)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="학습된 실험 이름 (outputs/ 아래)")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val")
    ap.add_argument("--conf", type=float, default=0.5, help="이 확신도 이상 예측만 근거로 씀")
    ap.add_argument("--draw", action="store_true", help="후보 사진을 그림으로 저장")
    ap.add_argument("--ocr", action="store_true", help="각인 OCR로 클래스 비교 (D11)")
    ap.add_argument(
        "--ocr-margin", type=float, default=0.3, help="다른 알약이 이만큼 더 비슷하면 후보"
    )
    args = ap.parse_args()

    cfg, model = load_experiment(args.exp)
    d = cfg["data"]
    records, _ = load_records(d["train_annotations"])
    if args.split != "all":
        records = load_split(records, f"{d['split_dir']}/{d['version']}/{args.split}.txt")
    class_map = load_class_map(d["class_map"])
    c2l, l2c = category_to_label(class_map), label_to_category(class_map)
    image_dir = resolve_path(d["train_images"])

    ocr = table = None
    if args.ocr:
        from src.ocr import PillOCR, imprint_table, match

        ocr = PillOCR()
        table = imprint_table(class_map, resolve_path(d["class_map"]).parent / "drug_info.json")
        print(f"각인 정보가 있는 알약 {len(table)}종 / 전체 {len(class_map)}종")

    out_dir = OUTPUT_DIR / args.exp
    draw_dir = out_dir / "label_issues"
    if args.draw:
        draw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    preds = model.predict([image_dir / r["file_name"] for r in records], conf_thr=0.05)
    for r, p in zip(records, preds, strict=True):
        gt = torch.tensor(r["boxes"], dtype=torch.float32).reshape(-1, 4)
        gt_lbl = torch.tensor([c2l[c] for c in r["category_ids"]], dtype=torch.long)
        keep = p["scores"] >= args.conf
        pb, ps, pl = p["boxes"][keep], p["scores"][keep], p["labels"][keep]
        iou = box_iou(gt, pb) if len(gt) and len(pb) else torch.zeros((len(gt), len(pb)))
        found = []
        for i, cid in enumerate(r["category_ids"]):
            base = {"file_name": r["file_name"], "category_id": cid, "gt_box": _fmt(gt[i])}
            same = (pl == gt_lbl[i]).nonzero().flatten().tolist()
            other = (pl != gt_lbl[i]).nonzero().flatten().tolist()
            best_same = max(same, key=lambda j: iou[i, j], default=None)
            best_other = max(other, key=lambda j: iou[i, j], default=None)
            if best_same is not None and iou[i, best_same] >= 0.75:
                pass  # 정상: 같은 알약을 정확한 자리에서 찾음
            elif best_same is not None and iou[i, best_same] >= 0.5:
                j = best_same
                rows.append(
                    {
                        **base,
                        "action": "box",
                        "new_value": _fmt(pb[j]),
                        "issue": "loose_box",
                        "pred_category_id": cid,
                        "score": f"{ps[j]:.3f}",
                        "iou": f"{iou[i, j]:.3f}",
                        "pred_box": _fmt(pb[j]),
                    }
                )
            elif best_other is not None and iou[i, best_other] >= 0.7:
                j = best_other
                rows.append(
                    {
                        **base,
                        "action": "class",
                        "new_value": l2c[int(pl[j])],
                        "issue": "wrong_class",
                        "pred_category_id": l2c[int(pl[j])],
                        "score": f"{ps[j]:.3f}",
                        "iou": f"{iou[i, j]:.3f}",
                        "pred_box": _fmt(pb[j]),
                    }
                )
            elif not len(p["boxes"]) or float(box_iou(gt[i : i + 1], p["boxes"]).max()) < 0.5:
                # 낮은 확신도 예측까지 봐도 이 자리에 아무것도 없음
                rows.append({**base, "action": "check", "new_value": "", "issue": "not_found"})
            found += [j for j in (best_same, best_other) if j is not None and iou[i, j] >= 0.3]
        for j in range(len(pb)):  # 어떤 정답과도 겹치지 않는 확신 높은 예측
            if j not in found and (len(gt) == 0 or float(iou[:, j].max()) < 0.3):
                rows.append(
                    {
                        "file_name": r["file_name"],
                        "category_id": l2c[int(pl[j])],
                        "action": "add",
                        "new_value": _fmt(pb[j]),
                        "issue": "missing_label",
                        "pred_category_id": l2c[int(pl[j])],
                        "score": f"{ps[j]:.3f}",
                        "pred_box": _fmt(pb[j]),
                    }
                )
        img = None
        if ocr is not None:
            img = read_image_rgb(image_dir / r["file_name"])
            for i, cid in enumerate(r["category_ids"]):
                text, _ = ocr.read(img, gt[i].tolist())
                ranked = match(text, table)
                if not ranked:
                    continue
                own = dict(ranked).get(int(gt_lbl[i]), 0.0)
                best_label, best_sim = ranked[0]
                if best_label != int(gt_lbl[i]) and best_sim - own >= args.ocr_margin:
                    rows.append(
                        {
                            "file_name": r["file_name"],
                            "category_id": cid,
                            "action": "class",
                            "new_value": l2c[best_label],
                            "issue": "ocr_mismatch",
                            "pred_category_id": l2c[best_label],
                            "score": f"{best_sim:.3f}",
                            "gt_box": _fmt(gt[i]),
                            "ocr_text": text,
                        }
                    )
        if args.draw and any(row["file_name"] == r["file_name"] for row in rows):
            img = read_image_rgb(image_dir / r["file_name"]) if img is None else img
            canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for b, lbl in zip(gt.tolist(), gt_lbl.tolist(), strict=True):
                cv2.rectangle(
                    canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 3
                )
                cv2.putText(canvas, f"GT {lbl}", (int(b[0]), int(b[1]) - 6), 0, 0.8, (0, 200, 0), 2)
            for b, lbl in zip(pb.tolist(), pl.tolist(), strict=True):
                cv2.rectangle(
                    canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 230), 2
                )
                cv2.putText(canvas, f"P {lbl}", (int(b[0]), int(b[3]) + 22), 0, 0.8, (0, 0, 230), 2)
            ok, buf = cv2.imencode(".jpg", canvas)
            buf.tofile(str(draw_dir / (r["file_name"].rsplit(".", 1)[0] + ".jpg")))

    out = out_dir / f"label_issues_{args.split}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:  # 엑셀에서 바로 열림
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["issue"]] = counts.get(row["issue"], 0) + 1
    print(f"이미지 {len(records)}장 확인, 후보 {len(rows)}건: {counts or '없음'}")
    print(f"저장: {out}" + (f", 그림: {draw_dir}" if args.draw else ""))
    print(
        "확인 후 맞는 줄의 앞 네 칸(file_name,category_id,action,new_value)만 manual_fix.csv에 옮기세요."
    )


if __name__ == "__main__":
    main()
```

**② `scripts/clean_labels.py` 전체 바꾸기**: `manual_fix.csv`를 읽어 고치는 부분(`load_manual_fix`, `apply_fixes`, `clean`의 앞부분, `main`의 검사)이 늘었습니다. D2·D5까지 병합된 상태라면 파일 전체를 아래로 바꿔도 다른 기능은 그대로입니다(`Ctrl+A` → 붙여넣기 → 저장).

```python
"""라벨 정제: 원본 어노테이션을 규칙대로 고쳐 새 데이터 버전(v1, v2 ...)을 만든다 (데이터 B).

실행:
    uv run python scripts/clean_labels.py --config configs/base.yaml --version v1

결과:
    data/annotations/v1.json        정제된 어노테이션 (COCO 형식 JSON 하나) — 커밋하지 않음 (대회 데이터 재배포 금지)
    data/annotations/v1_report.csv  무엇을 고쳤는지 기록 (보고서용) — 커밋
    data/splits/v1/                 v0 분할을 그대로 복사 (검증셋 고정) — 커밋
팀원은 이 스크립트와 manual_drop.csv를 받아 같은 명령을 실행하면 똑같은 v1.json이 만들어진다.

사용: 설정 파일에서 아래 두 줄만 바꾸면 모델이 정제 데이터로 학습한다.
    data:
      train_annotations: data/annotations/v1.json
      version: v1

규칙을 바꾸거나 추가하려면 clean() 함수만 고친다. 사람이 직접 확인한 오류는
data/annotations/manual_drop.csv (file_name,category_id) 에 적으면 해당 박스를 뺀다.
빼지 않고 고치려면 data/annotations/manual_fix.csv (실험 D10) 에 적는다:
    file_name,category_id,action,new_value
    a.png,1899,class,3482                  ← 1899로 잘못 적힌 박스를 3482로
    a.png,3482,box,100.0 120.0 260.0 300.0  ← 박스 좌표를 x1 y1 x2 y2(픽셀)로 바꿈
    b.png,1899,add,40.0 50.0 200.0 210.0    ← 빠진 알약 박스 추가
고치기가 먼저 적용되고, manual_drop은 고친 뒤의 category_id 기준으로 뺀다.
"""

from __future__ import annotations

import argparse
import csv
import shutil

from src.dataset import load_records
from src.dataset import records_to_coco as to_coco
from src.interfaces import MAX_PILLS_PER_IMAGE, Record
from src.utils import load_config, resolve_path, save_json

MIN_SIZE = 3  # 너비·높이가 이 픽셀보다 작으면 잘못된 박스로 본다


def load_manual_drop(path) -> set[tuple[str, int | None]]:
    """(file_name, category_id) 목록. category_id를 all 로 적으면 그 이미지 전체를 뺀다
    (라벨이 빠진 알약이 있는 이미지처럼 박스 하나로는 고칠 수 없는 경우)."""
    if not path.exists():
        return set()
    out: set[tuple[str, int | None]] = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = (row.get("category_id") or "").strip()
            out.add((row["file_name"].strip(), None if cid.lower() in ("", "all") else int(cid)))
    return out


def load_manual_fix(path) -> list[dict]:
    """manual_fix.csv 의 줄 목록 (실험 D10). 파일이 없으면 빈 목록."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return [
            {k: (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(f)
            if (row.get("file_name") or "").strip()
        ]


def apply_fixes(r: Record, fixes: list[dict], log: list[dict]) -> Record:
    """한 이미지에 manual_fix 줄을 적용한 사본을 돌려준다 (실험 D10)."""
    boxes = [list(b) for b in r["boxes"]]
    cats = list(r["category_ids"])
    for fx in fixes:
        cid, action, value = int(fx["category_id"]), fx["action"], fx.get("new_value", "")
        idx = [i for i, c in enumerate(cats) if c == cid]
        if action == "add":
            boxes.append([float(v) for v in value.split()])
            cats.append(cid)
        elif not idx:
            action = "not_found"  # 그 클래스 박스가 없음 → 기록만 (fix_not_found)
        elif action == "class":
            for i in idx:
                cats[i] = int(value)
        elif action == "box":
            boxes[idx[0]] = [float(v) for v in value.split()]
        else:
            continue  # check 등 고치지 않는 줄
        log.append({"file_name": r["file_name"], "category_id": cid, "action": f"fix_{action}"})
    return {**r, "boxes": boxes, "category_ids": cats}


def clean(
    records: list[Record], manual_drop: set, manual_fix: list[dict] | None = None
) -> tuple[list[Record], list[dict]]:
    """규칙대로 박스를 고치거나 빼고, 바꾼 내용을 log로 남긴다."""
    cleaned, log = [], []
    fixes_by_file: dict[str, list[dict]] = {}
    for fx in manual_fix or []:
        fixes_by_file.setdefault(fx["file_name"], []).append(fx)
    for r in records:
        if r["file_name"] in fixes_by_file:  # 실험 D10: 사람이 확인한 고침 먼저 적용
            r = apply_fixes(r, fixes_by_file[r["file_name"]], log)
        if (r["file_name"], None) in manual_drop:  # 이미지 전체 제외
            log.append(
                {"file_name": r["file_name"], "category_id": None, "action": "drop_image_manual"}
            )
            continue
        w, h = r["width"], r["height"]
        boxes, cats = [], []
        for (x1, y1, x2, y2), cid in zip(r["boxes"], r["category_ids"], strict=True):
            if (r["file_name"], cid) in manual_drop:
                log.append(
                    {"file_name": r["file_name"], "category_id": cid, "action": "drop_manual"}
                )
                continue
            # 규칙 1: 이미지 밖으로 나간 부분은 이미지 경계로 자른다
            nx1, ny1 = max(0.0, x1), max(0.0, y1)
            nx2, ny2 = (min(float(w), x2) if w else x2), (min(float(h), y2) if h else y2)
            if (nx1, ny1, nx2, ny2) != (x1, y1, x2, y2):
                log.append({"file_name": r["file_name"], "category_id": cid, "action": "clip"})
            # 규칙 2: 너무 작거나 넓이가 없는 박스는 뺀다
            if nx2 - nx1 < MIN_SIZE or ny2 - ny1 < MIN_SIZE:
                log.append({"file_name": r["file_name"], "category_id": cid, "action": "drop_tiny"})
                continue
            boxes.append([nx1, ny1, nx2, ny2])
            cats.append(cid)
        # 규칙 3: 과제 조건(최대 4개)을 넘으면 기록만 남긴다 (자동 삭제하지 않고 사람이 확인)
        if len(boxes) > MAX_PILLS_PER_IMAGE:
            log.append(
                {"file_name": r["file_name"], "category_id": None, "action": "check_too_many"}
            )
        # 규칙 4: 박스가 하나도 없는 이미지는 학습에서 뺀다
        if not boxes:
            log.append(
                {"file_name": r["file_name"], "category_id": None, "action": "drop_image_no_box"}
            )
            continue
        cleaned.append({**r, "boxes": boxes, "category_ids": cats})
    return cleaned, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--version", default="v1")
    ap.add_argument(
        "--source", default=None, help="원본 어노테이션 (기본: config의 train_annotations)"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]

    records, categories = load_records(args.source or d["train_annotations"])
    ann_dir = resolve_path(d["class_map"]).parent / "annotations"  # 기본: data/annotations
    manual_fix = load_manual_fix(ann_dir / "manual_fix.csv")
    unknown = {
        int(v)
        for fx in manual_fix
        for v in ([fx["category_id"]] + ([fx["new_value"]] if fx["action"] == "class" else []))
        if int(v) not in categories
    }
    if unknown:
        raise SystemExit(
            f"manual_fix.csv에 학습 데이터에 없는 category_id가 있습니다: {sorted(unknown)}"
        )
    cleaned, log = clean(records, load_manual_drop(ann_dir / "manual_drop.csv"), manual_fix)

    save_json(to_coco(cleaned, categories), ann_dir / f"{args.version}.json")
    with open(ann_dir / f"{args.version}_report.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "category_id", "action"])
        writer.writeheader()
        writer.writerows(log)

    # 검증셋 고정: v0 분할을 새 버전으로 복사 (정제로 빠진 이미지는 학습 때 자동 제외됨)
    src_split = resolve_path(d["split_dir"]) / "v0"
    dst_split = resolve_path(d["split_dir"]) / args.version
    if src_split.exists() and not dst_split.exists():
        shutil.copytree(src_split, dst_split)

    actions: dict[str, int] = {}
    for row in log:
        actions[row["action"]] = actions.get(row["action"], 0) + 1
    print(f"원본 {len(records)}장 → 정제 {len(cleaned)}장, 수정 내역: {actions or '없음'}")
    print(
        f"저장: {ann_dir / (args.version + '.json')}, {args.version}_report.csv, splits/{args.version}/"
    )


if __name__ == "__main__":
    main()
```

**③ 새 파일 `data/annotations/manual_fix.csv`** (머리줄 한 줄만. 이 파일은 커밋합니다)

```csv
file_name,category_id,action,new_value
```

| 칸 | 값 |
| --- | --- |
| `file_name` | 이미지 파일 이름 |
| `category_id` | 고칠 박스의 **지금** 클래스 번호 (`add`면 추가할 클래스) |
| `action` | `class` 클래스 바꾸기 / `box` 좌표 바꾸기 / `add` 빠진 박스 추가 / `check` 기록만 |
| `new_value` | `class`면 새 category\_id, `box`·`add`면 `x1 y1 x2 y2` (픽셀, 공백으로 구분) |

한 이미지에 같은 클래스 박스가 둘 이상이면 `class`는 모두 바꾸고 `box`는 첫 번째만 바꿉니다. 그런 이미지는 `manual_drop.csv`로 빼는 편이 안전합니다.

**④ 코드 확인**

```bash
uv run python scripts/find_label_issues.py --help
uv run ruff check . && uv run pytest -q      # 테스트를 추가했다면 개수가 늘어남
```

### 실행

```bash
# 1) 후보 찾기 (학습된 baseline 실험 사용, 검증셋부터)
uv run python scripts/find_label_issues.py --exp 1028_modelA_yolov8s_v0_baseline --split val --draw
uv run python scripts/find_label_issues.py --exp 1028_modelA_yolov8s_v0_baseline --split train --draw
# 2) outputs/<실험>/label_issues_val.csv 를 엑셀로 열고, outputs/<실험>/label_issues/ 의 그림을 함께 확인
# 3) 맞는 줄의 앞 네 칸만 data/annotations/manual_fix.csv 에 복사
# 4) 고친 데이터 버전 만들기 → 비교
uv run python scripts/clean_labels.py --config configs/base.yaml --version v1
uv run python scripts/evaluate.py --exp 1028_modelA_yolov8s_v0_baseline --val-config configs/yolo_v1_clean.yaml
```

**그림 읽는 법**: 초록 박스 `GT 번호`가 지금 라벨, 빨강 박스 `P 번호`가 모델 예측입니다(번호는 label\_idx, `data/class_map.json`에서 약 이름 확인). 그림은 대회 이미지이므로 PR·디스코드에 올리지 않습니다.

합성 연습 데이터에 일부러 오류 3개(클래스 바꾸기, 박스 15픽셀 넓히기, 박스 하나 지우기)를 넣고 연습 모델(검증 mAP 0.21)로 돌린 결과입니다.

```text
이미지 16장 확인, 후보 15건: {'not_found': 10, 'missing_label': 2, 'loose_box': 3}
```

- 지운 박스는 `missing_label`, 넓힌 박스는 `not_found`로 잡혔습니다. 클래스를 바꾼 박스는 연습 모델의 확신도(0.07)가 낮아 잡히지 않았습니다.
- 약한 모델일수록 `not_found`가 많아집니다. **가장 좋은 모델로 돌리고**, 연습 모델처럼 확신도가 낮으면 `--conf 0.3`으로 낮춥니다.
- 후보 줄을 `manual_fix.csv`에 옮겨 다시 만들면 `수정 내역: {'fix_class': 1, 'fix_box': 1, 'fix_add': 1}`처럼 고친 수가 나옵니다.

### 커밋할 것 / 하지 않을 것

| 커밋 | 커밋하지 않음 |
| --- | --- |
| `scripts/find_label_issues.py`, `scripts/clean_labels.py`, `data/annotations/manual_fix.csv`, `data/annotations/v1_report.csv` | `outputs/<실험>/label_issues*` (후보 목록·그림), `data/annotations/v1.json` |

고친 이유는 보고서 재료입니다. PR 본문에 “검증셋 후보 N건 중 M건 수정(클래스 a, 박스 b, 추가 c)”과 점수 변화를 적습니다.

## D11. 각인(식별문자) OCR 활용

**한 줄 요약**: 알약에는 앞뒤로 글자가 새겨져 있고, 약 정보표(`data/drug_info.json`)에 `식별문자(앞)`, `식별문자(뒤)`가 들어 있습니다. 모양·색이 같은 알약끼리는 이 글자가 결정적인 단서입니다. 박스를 잘라 OCR로 읽고 각인 표와 비교해 ① **라벨 점검**(D10 `--ocr`)과 ② **확신도가 낮은 예측의 이름 다시 고르기**(`ocr_rerank`)에 씁니다.

**현실적인 기대치**: 각인은 작고 음각이며 빛이 반사되어 OCR이 못 읽는 경우가 많습니다. 그래서 **읽은 경우에만, 각인이 아주 비슷할 때만** 이름을 바꾸고, 못 읽으면 원래 예측을 그대로 둡니다. 박스 위치는 바꾸지 않으므로 이름이 틀렸던 박스를 바로잡는 만큼만 점수가 오릅니다.

**OCR 엔진**: RapidOCR(`rapidocr-onnxruntime`). pip 패키지 안에 모델이 들어 있어 따로 내려받지 않고, 프로젝트가 쓰는 `opencv-python`과 충돌하지 않습니다. 박스를 4방향으로 돌려 가며 읽고 가장 확신하는 결과를 씁니다.

### 코드 만들기

**① 패키지 추가 (Experimentation Lead)**: `uv.lock`도 함께 바뀌므로 한 사람이 하고 PR로 올립니다.

```bash
uv add --optional ocr rapidocr-onnxruntime
# pyproject.toml 의 [project.optional-dependencies] 에 ocr = ["rapidocr-onnxruntime>=…"] 가 생김
# 팀원: git pull 후 uv sync --extra demo --extra ocr
```

**② 새 파일 `src/ocr.py`** (데이터 A, 전체를 그대로 복사)

```python
"""알약 각인(식별문자) OCR 도우미 (실험 D11, 데이터 A).

알약 사진을 잘라 글자를 읽고, 약 정보표(data/drug_info.json)의 식별문자(앞/뒤)와 비교해
어느 알약과 가장 비슷한지 점수를 매긴다. 라벨 점검(find_label_issues.py --ocr)과
예측 이름 다시 고르기(src/models/ocr_rerank.py)가 함께 쓴다.

OCR 엔진: RapidOCR (pip 패키지 안에 모델이 들어 있어 따로 내려받지 않음)
    설치: uv sync --extra ocr
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import cv2
import numpy as np

from src.interfaces import ClassInfo
from src.utils import load_json, resolve_path

FRONT, BACK = "식별문자(앞)", "식별문자(뒤)"  # build_drug_info.py가 만든 항목 이름


def normalize(text: str | None) -> str:
    """대소문자·공백·기호를 무시하고 영문 대문자와 숫자만 남긴다 (예: 'ty 500' → 'TY500')."""
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def imprint_table(class_map: list[ClassInfo], drug_info_path) -> dict[int, list[str]]:
    """label_idx → 비교할 각인 문자열 목록 (앞, 뒤, 앞+뒤, 뒤+앞). 각인이 없는 알약은 빠진다."""
    path = resolve_path(drug_info_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 이 없습니다. 먼저 scripts/build_drug_info.py를 실행하세요."
        )
    info = load_json(path)
    table: dict[int, list[str]] = {}
    for c in class_map:
        row = info.get(str(c["category_id"]), {})
        front, back = normalize(row.get(FRONT)), normalize(row.get(BACK))
        options = {s for s in (front, back, front + back, back + front) if len(s) >= 2}
        if options:
            table[c["label_idx"]] = sorted(options)
    return table


def match(text: str, table: dict[int, list[str]]) -> list[tuple[int, float]]:
    """읽은 글자와 각 알약 각인의 비슷한 정도(0~1)를 높은 순서로."""
    if len(text) < 2:
        return []
    scores = [
        (label, max(SequenceMatcher(None, text, s).ratio() for s in options))
        for label, options in table.items()
    ]
    return sorted(scores, key=lambda x: -x[1])


class PillOCR:
    """알약 하나를 잘라 4방향으로 돌려 읽고, 가장 확신하는 결과의 글자를 돌려준다."""

    def __init__(self, min_conf: float = 0.5, min_side: int = 160):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise ImportError("OCR 패키지가 없습니다. 실행: uv sync --extra ocr") from e
        self.engine = RapidOCR()
        self.min_conf = min_conf
        self.min_side = min_side

    def crop(self, img: np.ndarray, box, pad: float = 0.05) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box)
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = max(0, int(x1 - pad * bw)), min(w, int(x2 + pad * bw))
        y1, y2 = max(0, int(y1 - pad * bh)), min(h, int(y2 + pad * bh))
        c = img[y1:y2, x1:x2]
        if c.size == 0:
            return c
        scale = max(1.0, self.min_side / min(c.shape[:2]))  # 작은 알약은 키워서 읽는다
        return cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def read(self, img: np.ndarray, box) -> tuple[str, float]:
        """(정리된 글자, 확신도). 못 읽으면 ('', 0)."""
        c = self.crop(img, box)
        if c.size == 0:
            return "", 0.0
        best = ("", 0.0)
        for k in range(4):
            result, _ = self.engine(np.ascontiguousarray(np.rot90(c, k)))
            if not result:
                continue
            parts = [(normalize(r[1]), float(r[2])) for r in result if float(r[2]) >= self.min_conf]
            text = "".join(t for t, _ in parts)
            conf = sum(s for _, s in parts) / len(parts) if parts else 0.0
            if len(text) >= 2 and conf > best[1]:
                best = (text, conf)
        return best
```

**③ 새 파일 `src/models/ocr_rerank.py`** (데이터 A, 전체를 그대로 복사): 이미 학습된 실험의 예측을 받아 확신도가 낮은 박스만 각인으로 다시 확인합니다. 모델 가이드 M4(2단계)와 같은 방식이라 평가·제출·데모에서 실험 이름으로 그대로 쓸 수 있습니다.

```python
"""탐지 + 각인 OCR로 알약 이름 다시 고르기 (실험 D11, 데이터 A·모델 B).

1단계: 이미 학습이 끝난 실험(model.detector_exp: YOLO, R-CNN, two_stage 모두 가능)이 박스를 찾는다.
2단계: 확신도가 낮은 박스(score < ocr_max_score)만 잘라 각인을 읽고, 약 정보표의 식별문자와 비교한다.
  - 읽은 글자가 어떤 알약과 ocr_min_sim 이상 비슷하면 그 알약으로 이름을 정한다.
  - 점수 = (1 - ocr_weight) x 탐지 점수 + ocr_weight x 각인 비슷한 정도
  - 글자를 못 읽었거나 비슷한 알약이 없으면 1단계 결과를 그대로 둔다.
박스 위치는 바꾸지 않는다. 학습은 하지 않으므로 run_train.py는 설정 저장과 검증 채점만 한다.

필요: uv sync --extra ocr, data/drug_info.json (scripts/build_drug_info.py)
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.dataset import read_image_rgb
from src.interfaces import Prediction
from src.models.base import BaseDetector
from src.utils import load_class_map, resolve_path, save_json


class OcrRerankDetector(BaseDetector):
    name = "ocr_rerank"

    def __init__(self, cfg: dict, num_classes: int):
        super().__init__(cfg, num_classes)
        self.detector = None
        self.ocr = None
        self.table: dict[int, list[str]] = {}

    def fit(self, out_dir: Path) -> Path:
        # 학습할 것이 없다. 어떤 탐지 실험과 묶였는지만 best.pt 자리에 기록한다
        best = out_dir / "best.pt"
        save_json({"detector_exp": self.cfg["model"]["detector_exp"]}, best)
        return best

    def load(self, weights: str | Path) -> None:
        from src.infer import load_experiment  # 실험 폴더에서 1단계 모델을 읽는다
        from src.ocr import PillOCR, imprint_table

        m, d = self.cfg["model"], self.cfg["data"]
        _, self.detector = load_experiment(m["detector_exp"])
        if self.detector.num_classes != self.num_classes:
            raise ValueError(
                "탐지 실험과 클래스 수가 다릅니다. 같은 class_map으로 학습한 실험을 쓰세요."
            )
        self.ocr = PillOCR(min_conf=float(m.get("ocr_min_conf", 0.5)))
        drug_info = m.get("drug_info") or resolve_path(d["class_map"]).parent / "drug_info.json"
        self.table = imprint_table(load_class_map(d["class_map"]), drug_info)

    @torch.no_grad()
    def predict(self, image_paths: list[Path], conf_thr: float | None = None) -> list[Prediction]:
        from src.ocr import match

        m = self.cfg["model"]
        max_score = float(m.get("ocr_max_score", 0.6))
        min_sim = float(m.get("ocr_min_sim", 0.8))
        w = float(m.get("ocr_weight", 0.5))
        self.detector.cfg["eval"].update(self.cfg["eval"])  # --tta 등을 1단계에도 적용
        out = []
        for path, p in zip(image_paths, self.detector.predict(image_paths, conf_thr), strict=True):
            todo = (p["scores"] < max_score).nonzero().flatten().tolist()
            if not todo:
                out.append(p)
                continue
            img = read_image_rgb(path)
            scores, labels = p["scores"].clone(), p["labels"].clone()
            for i in todo:
                text, _ = self.ocr.read(img, p["boxes"][i].tolist())
                ranked = match(text, self.table)
                if ranked and ranked[0][1] >= min_sim:
                    labels[i] = ranked[0][0]
                    scores[i] = (1 - w) * scores[i] + w * ranked[0][1]
            out.append(Prediction(**{**p, "scores": scores, "labels": labels}))
        return out
```

**④ 모델 등록 `src/models/__init__.py`** (Experimentation Lead 승인): 마지막 `raise ValueError(...)` 줄 바로 위에 넣고 `raise` 줄을 바꿉니다.

```python
    if family == "ocr_rerank":  # 실험 D11: 각인 OCR로 이름 다시 고르기
        from src.models.ocr_rerank import OcrRerankDetector

        return OcrRerankDetector(cfg, num_classes)
    raise ValueError(f"알 수 없는 model.family: {family} (yolo / rcnn / two_stage / ocr_rerank)")
```

(모델 가이드 M4를 아직 안 했다면 `two_stage` 줄은 없어도 됩니다.)

**⑤ 새 설정 `configs/ocr_rerank.yaml`**

```yaml
# 실험 D11 (데이터·모델): 탐지 결과 중 확신도가 낮은 박스를 각인 OCR로 다시 확인해 이름을 고른다
# 먼저: uv sync --extra ocr, scripts/build_drug_info.py (data/drug_info.json), 탐지 실험이 outputs/ 에 있어야 함
base: yolo_baseline.yaml
exp_name: 1103_dataA_ocr_v0_rerank
owner: dataA
note: YOLO baseline + 각인 OCR 재분류 (D11)
model:
  family: ocr_rerank
  detector_exp: 1028_modelA_yolov8s_v0_baseline   # ← 내 PC의 최고 탐지 실험 이름으로 바꾼다
  ocr_max_score: 0.6          # 이 점수보다 낮은 박스만 OCR로 확인 (1.0이면 모든 박스)
  ocr_min_sim: 0.8            # 각인이 이만큼 비슷해야 이름을 바꿈 (0~1)
  ocr_weight: 0.5             # 새 점수에서 각인 비슷한 정도의 비중
  ocr_min_conf: 0.5           # OCR이 이 확신도 이상으로 읽은 글자만 씀
```

**⑥ 코드 확인**: 각인 표와 OCR 엔진이 동작하는지 먼저 봅니다. 약 정보표에 각인이 있는 알약 수가 출력되면 됩니다.

```bash
uv run python scripts/build_drug_info.py --config configs/base.yaml     # data/drug_info.json (없을 때)
uv run python -c "from src.ocr import PillOCR, imprint_table; from src.utils import load_class_map; t = imprint_table(load_class_map('data/class_map.json'), 'data/drug_info.json'); print('각인 있는 알약', len(t)); PillOCR(); print('OCR 준비 완료')"
```

### 실행 ① 라벨 점검에 쓰기

```bash
uv run python scripts/find_label_issues.py --exp <최고 실험> --split all --ocr --draw
```

`ocr_mismatch` 줄은 “정답 라벨의 각인보다 다른 알약의 각인과 0.3 이상 더 비슷하게 읽힌 박스”입니다. 그림에서 글자를 직접 읽어 보고, 맞으면 D10처럼 `manual_fix.csv`에 옮깁니다. 각인 정보가 비어 있는 알약은 비교에서 빠집니다.

### 실행 ② 예측 이름 다시 고르기

```bash
# configs/ocr_rerank.yaml 의 detector_exp를 내 최고 실험 이름으로 바꾼 뒤
uv run python scripts/run_train.py --config configs/ocr_rerank.yaml        # 학습 없음, 설정 저장 + 검증 채점
uv run python scripts/evaluate.py --exp 1103_dataA_ocr_v0_rerank --class-metrics
uv run python scripts/make_submission.py --exp 1103_dataA_ocr_v0_rerank
```

**비교 기준은 `detector_exp`로 쓴 실험의 검증 점수**입니다. 값은 재학습 없이 `outputs/<실험>/config.yaml`만 고치고 `evaluate.py`를 다시 돌려 찾습니다.

| 설정 | 기본 | 올리면 | 내리면 |
| --- | --- | --- | --- |
| `ocr_max_score` | 0.6 | 더 많은 박스를 OCR로 확인 (느려짐) | 헷갈리는 박스만 확인 |
| `ocr_min_sim` | 0.8 | 확실할 때만 이름 변경 | 이름이 자주 바뀜 (오답 위험) |
| `ocr_weight` | 0.5 | 각인이 맞은 박스의 점수를 더 올림 | 탐지 점수를 더 믿음 |

**확인한 것 (합성 연습 데이터)**: 합성 알약에 `AB12`를 그려 넣고 각인 표에 같은 값을 넣었더니 `OCR: AB12 1.0 → 해당 알약 1위`로 맞게 골랐습니다. 연습 알약에는 각인이 없어 `ocr_rerank`는 이름을 바꾸지 않았고(검증 점수 0.21449 그대로), 평가와 제출 파일 만들기까지 오류 없이 끝났습니다. 실제 데이터에서 얼마나 읽히는지는 **실행 ①의 그림으로 먼저 확인**하고 ②를 진행합니다.

**속도**: CPU 기준 박스 하나에 약 0.3\~1초입니다. 테스트 842장 전체를 `ocr_max_score: 1.0`으로 돌리면 수십 분이 걸릴 수 있으니 기본값(0.6)부터 씁니다.

### 커밋할 것 / 하지 않을 것

| 커밋 | 커밋하지 않음 |
| --- | --- |
| `src/ocr.py`, `src/models/ocr_rerank.py`, `src/models/__init__.py`, `configs/ocr_rerank.yaml`, `pyproject.toml`, `uv.lock` | `data/drug_info.json` (대회 데이터에서 뽑은 정보), 잘라낸 알약 그림, `outputs/` |

## 실험 기록, 결과 판단, PR

**순서**: 하나씩 baseline과 비교 → 효과 있는 것만 남김 → 남은 것끼리 합친 실험을 한 번 더 합니다. 합칠 때는 “데이터 버전(v1→v2) + 증강 + 오버샘플링”을 한 설정 파일에 모으고, 이름에 모두 드러나게 적습니다(예: `1101_dataB_yolov8s_v2_augsafe_rfs01`).

**같은 기준으로 비교**: 빠른 비교표는 모델 가이드의 “따라하기 실습 Step 5” 명령(실험 기록표 출력)이나 `make_reports.py 03`을 씁니다. 검증 정답이 바뀐 실험(v1)은 D5의 `--val-config`로 baseline을 다시 채점해 비교합니다.

**판단 기준**: 검증셋이 작아서(수십 장) 점수가 우연히 흔들립니다. mAP@\[0.75:0.95\]가 0.01 이상 오르면 “효과 있음”, 그보다 작으면 “불분명”으로 기록하고, 불분명한 실험은 합치기 후보에서 뒤로 미립니다. 클래스별 효과가 궁금하면 `evaluate.py --class-metrics`로 드문 클래스 AP가 올랐는지 봅니다(D1·D2의 목표).

### 실험 기록표 (각자 채워 주세요)

| ID | 설정 파일 | 비교 기준 | mAP@\[0.75:0.95\] | 변화 | 판단 | 담당 |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | `yolo_rfs.yaml` | YOLO baseline |  |  |  |  |
| D1 | `rcnn_balance.yaml` | R-CNN baseline |  |  |  |  |
| D2 | `yolo_v2_copypaste.yaml` | YOLO baseline |  |  |  |  |
| D3 | `yolo_aug_safe.yaml` | YOLO baseline |  |  |  |  |
| D3 | `yolo_aug_rotate.yaml` | YOLO baseline |  |  |  |  |
| D3 | `rcnn_aug.yaml` | R-CNN baseline |  |  |  |  |
| D4 | `yolo_img1024.yaml` | YOLO baseline |  |  |  |  |
| D5 | `yolo_v1_clean.yaml` | YOLO baseline (`--val-config` v1) |  |  |  |  |
| D6 | `yolo_v3_aihub.yaml` | YOLO baseline |  |  |  |  |
| D6 | `yolo_v3all_aihub.yaml` | Kaggle 점수 (검증은 하락 여부만) |  |  |  |  |
| D10 | `yolo_v1_clean.yaml` (manual\_fix 적용) | YOLO baseline (`--val-config` v1) |  |  |  |  |
| D11 | `ocr_rerank.yaml` | `detector_exp`로 쓴 실험 |  |  |  |  |

### PR에 담을 것

- 단계별로 PR 하나: 새로 만든 설정 파일과 고친 코드 파일(D1 `src/dataset.py`·`src/utils.py`·`scripts/export_yolo.py`·`scripts/check_data.py`·`src/models/yolo.py`, D2 `scripts/make_copy_paste.py`·`src/dataset.py`·`scripts/clean_labels.py`, D3 `src/transforms.py`, D6 `scripts/make_external.py`·`.gitignore`, D10 `scripts/find_label_issues.py`·`scripts/clean_labels.py`, D11 `src/ocr.py`·`src/models/ocr_rerank.py`, D5 `scripts/clean_labels.py`), 그리고 결과 기록 파일 `manual_drop.csv`, `manual_fix.csv`, `*_report.csv`, `*_copy_paste.csv`, `splits/v*/`
- PR 전에 `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q`가 모두 통과해야 합니다
- PR 본문: 위 표의 해당 행(숫자)과 판단 한 줄
- 넣지 않을 것: `data/synth/`, `data/annotations/*.json`, 학습 그림(`train_batch*.jpg`), HTML 리포트 — 모두 대회 이미지라 공개 레포에 올리지 않습니다(`.gitignore`가 대부분 막음)

점수가 오르지 않은 실험도 PR로 남깁니다. “임의 회전은 박스가 헐거워져 오히려 떨어졌다” 같은 결과는 보고서의 좋은 재료입니다.
