# 모델 개선 실험 가이드 (v2)

Oct 24, 2026 · @Doosung Bang

이 문서는 [모델 개선 담당 실행 가이드](https://claude.ai/artifact/UWzRrEJm5qtLXsrFZ4XcpR)(baseline 버전)의 따라하기 실습을 끝낸 뒤 점수를 올리기 위한 **모델 쪽 개선 실험 모음**입니다. v1 실습을 마친 main(baseline 고정)에서 출발해, 버그 수정 → 새 설정 파일 만들기 → 코드 수정 → 결과 확인까지 이 문서만 보고 따라 할 수 있게 적었습니다. 새 기능은 기본값이 모두 꺼져 있어 baseline 점수는 바뀌지 않습니다.

## 시작 전에: 실험 목록

**이번 지표에서 점수가 나오는 곳**: mAP@\[0.75:0.95\]는 ① 클래스별 점수의 평균이고 ② 박스가 정답과 75\~95% 겹쳐야 맞은 것으로 줍니다. 그래서 **드문 클래스를 놓치지 않기**와 **박스 위치를 정확하게**가 두 축입니다.

| ID | 실험 | 바꾸는 곳 | 재학습 | 우선 |
| --- | --- | --- | --- | --- |
| T2 | 드문 클래스 손실 가중치 / 오버샘플링 | `yolo_clspw.yaml`, `rcnn_balance.yaml` | 필요 | 1 |
| M1 | 모델 크기 s → m | `yolo_v8m.yaml` | 필요 | 1 |
| M2 | 최신 버전 YOLO11 / YOLO26 | `yolo11s.yaml`, `yolo26s.yaml` | 필요 | 1 |
| M3 | 위치(박스) 손실 비중 키우기 | `yolo_box_gain.yaml` | 필요 | 2 |
| T1 | 더 오래, 학습률·스케줄 지정 | `yolo_long.yaml` | 필요 | 2 |
| R1 | R-CNN 최고 epoch 저장 | `rcnn.py`, `rcnn_select_best.yaml` | 필요 | 2 |
| P1 | 확신도·NMS 설정 찾기 | `evaluate.py --nms-iou` | 없음 | 1 |
| P4 | 예측 TTA | `evaluate.py --tta`, `make_submission.py --tta` | 없음 | 1 |
| P2 | WBF 앙상블 | `--ensemble … --method wbf` | 없음 | 2 |
| M4 | 탐지 + 분류 2단계 | 새 모델 파일 | 필요 | 3 |

우선 1은 첫째 주, 2는 첫 결과를 본 뒤, 3은 시간이 남을 때입니다. **재학습이 없는 P1·P4·P2는 몇 분이면 끝나므로, 학습이 끝난 모든 실험에 바로 적용해 봅니다.** 데이터 쪽 개선(오버샘플링, 합성, 증강, 해상도)은 [데이터 개선 실험 가이드 (v2)](https://claude.ai/artifact/TK8QJtrqp79hK7oqFt7yfH)에 있습니다.

**준비**: v1 가이드 실습이 모두 main에 병합된 상태(= baseline 고정)에서 시작합니다. `git switch <내이름>` → `git pull origin main` → `uv sync --extra demo` → `uv run pytest -q`(**4 passed**)로 출발점을 맞춥니다. 이 문서에 나오는 설정 파일과 코드는 아직 레포에 없고, 각 담당이 아래 단계대로 직접 만듭니다. 설정 파일의 `exp_name` 앞 날짜와 `owner`는 실제 날짜와 내 이름으로 바꿔 씁니다.

**규칙**: 한 실험에 하나만 바꾸고, 학습 전에 `--epochs 1 --exp-name test_…`로 설정 오류부터 확인합니다(baseline 가이드 실습 Step 3).

## 0. 순서와 담당, 그리고 버그 수정(M0)

같은 파일을 두 사람이 동시에 고치면 PR이 충돌합니다. 아래 순서를 지키고, 앞 단계 PR이 main에 병합되면 `git pull origin main`을 한 뒤 다음 단계를 시작합니다. 같은 순서 번호끼리는 동시에 해도 됩니다.

| 순서 | 단계 | 담당 | 고치는 파일 |
| --- | --- | --- | --- |
| 1 | M0 버그 수정 | 모델 A | `src/models/yolo.py` |
| 2 | T2·M1\~M3·T1 설정 파일 만들기 | 모델 A (YOLO), 모델 B (R-CNN) | `configs/` 새 파일 |
| 2 | R1 R-CNN 최고 epoch 저장 | 모델 B | `src/models/rcnn.py`, 새 설정 파일 |
| 2 | P2 WBF 앙상블 함수 | 모델 B | `src/ensemble.py` |
| 3 | P4 예측 TTA (한 줄) | 모델 A | `src/models/yolo.py` |
| 4 | PS 평가·제출 스크립트 확장 | Experimentation Lead | `src/infer.py`, `scripts/evaluate.py`, `scripts/make_submission.py` |
| 5 | 테스트 추가 (선택) | Experimentation Lead | `tests/test_pipeline.py` |

데이터 B의 D1도 `src/models/yolo.py`의 한 줄을 고칩니다. 서로 다른 줄이라 보통 충돌하지 않지만, PR 화면에 충돌(conflict)이 뜨면 Experimentation Lead에게 알립니다. PS는 P2·P4가 병합된 뒤에 시작합니다(새 함수를 불러 쓰기 때문).

### M0. 버그 수정: 설정에 `plots: true`를 넣으면 학습이 시작되지 않음 (모델 A)

**증상**: `configs/yolo_aug_rotate.yaml`처럼 `yolo_args`에 `plots: true`가 있는 설정으로 학습하면 바로 멈춥니다.

```text
TypeError: ...train() got multiple values for keyword argument 'plots'
```

**원인**: 코드가 `plots=False`를 직접 넘기면서 `yolo_args`의 `plots`도 한 번 더 넘겨, 같은 이름이 두 번 들어갑니다. 고친 뒤에는 기본값을 사전(dict)에 먼저 담고 `yolo_args`로 덮어씁니다.

**① 찾기**: VS Code에서 `src/models/yolo.py`를 열고 `Ctrl+F`로 `self.model.train(`을 찾습니다. `fit` 함수 안의 아래 부분 전체를 선택합니다.

```python
        self.model.train(
            data=str(data_yaml),
            epochs=t["epochs"],
            imgsz=self.cfg["data"]["img_size"],
            batch=t["batch_size"],
            lr0=t["lr"],
            optimizer=t.get("optimizer", "auto"),
            workers=t.get("num_workers", 4),
            device=self._device_arg(),
            seed=self.cfg.get("seed", 42),
            project=str(out_dir),
            name="yolo_run",
            exist_ok=True,
            plots=False,
            verbose=False,
            **self.cfg.get("yolo_args", {}),  # ultralytics 고유 옵션은 configs의 yolo_args에
        )
```

**② 바꾸기**: 선택한 부분을 아래로 바꿔 붙여넣고 저장합니다(앞 공백 8칸 유지).

```python
        args = {
            "data": str(data_yaml),
            "epochs": t["epochs"],
            "imgsz": self.cfg["data"]["img_size"],
            "batch": t["batch_size"],
            "lr0": t["lr"],
            "optimizer": t.get("optimizer", "auto"),
            "workers": t.get("num_workers", 4),
            "device": self._device_arg(),
            "seed": self.cfg.get("seed", 42),
            "project": str(out_dir),
            "name": "yolo_run",
            "exist_ok": True,
            "plots": False,
            "verbose": False,
        }
        # ultralytics 고유 옵션은 configs의 yolo_args에. 같은 키(plots 등)는 yolo_args 값이 이긴다
        args.update(self.cfg.get("yolo_args") or {})
        self.model.train(**args)
```

**③ 확인**: 전에는 멈추던 설정으로 1 epoch만 돌려 봅니다. 마지막 줄에 검증 점수가 찍히고 `outputs/test_m0/yolo_run/` 안에 그림 파일(`train_batch0.jpg` 등)이 생기면 성공입니다. 그림은 대회 이미지라 커밋하지 않습니다.

```bash
uv run python scripts/run_train.py --config configs/yolo_aug_rotate.yaml --epochs 1 --exp-name test_m0
uv run ruff check . && uv run ruff format --check .
uv run pytest -q      # 4 passed
```

**④ PR**: `git add src/models/yolo.py` → `git commit -m "fix: yolo_args plots 중복 인자 오류 수정 (M0)"` → `git push` → PR. 이 PR이 병합되어야 다른 YOLO 실험(특히 D3의 `yolo_aug_safe.yaml`)이 돌아갑니다.

## T2. 클래스 불균형: 드문 클래스에 비중 더 주기

**한 줄 요약**: 흔한 알약만 잘 맞히는 모델을 막기 위해 드문 클래스의 비중을 키웁니다. YOLO는 **손실 가중치**, R-CNN은 **드문 이미지 더 자주 뽑기**로, 둘 다 설정 값 하나로 켭니다. 코드는 baseline에 이미 있으므로(R-CNN 샘플러는 rcnn.py, YOLO의 cls\_pw는 Ultralytics 기능) 설정 파일 두 개만 새로 만듭니다.

| 모델 | 설정 파일 | 켜는 값 | 방식 |
| --- | --- | --- | --- |
| YOLO | `configs/yolo_clspw.yaml` | `yolo_args: {cls_pw: 0.5}` | 클래스 빈도의 역수^0.5만큼 분류 손실을 키움 (Ultralytics 기본 기능) |
| Faster R-CNN | `configs/rcnn_balance.yaml` | `train: {balance_power: 0.5}` | 드문 클래스가 든 이미지를 학습에 더 자주 뽑음 (`rcnn.py`의 샘플러) |

**① 설정 파일 만들기**: VS Code 왼쪽 탐색기에서 `configs` 폴더를 오른쪽 클릭 → **New File** → 파일 이름을 입력하고, 아래 내용을 그대로 붙여넣어 저장합니다. YOLO 파일은 모델 A, R-CNN 파일은 모델 B가 만듭니다.

```yaml
# configs/yolo_clspw.yaml
# 실험 T2 (모델): 드문 클래스의 분류 손실을 키운다 (Ultralytics cls_pw, 학습 로그에 Class weights 출력)
base: yolo_baseline.yaml
exp_name: 1026_modelA_yolov8s_v0_clspw05
owner: modelA
note: 클래스 빈도 역수^0.5 손실 가중치 (T2)
yolo_args:
  cls_pw: 0.5
```

```yaml
# configs/rcnn_balance.yaml
# 실험 D1/T2 (R-CNN): 드문 클래스가 든 이미지를 더 자주 뽑는다 (rcnn.py _balanced_sampler)
base: rcnn_baseline.yaml
exp_name: 1026_modelB_frcnn_v0_balance05
owner: modelB
note: 드문 클래스 이미지 오버샘플링 power 0.5 (D1/T2)
train:
  balance_power: 0.5
```

**② 테스트 실행 → 본 학습**: 먼저 `--epochs 1 --exp-name test_t2`를 붙여 오류가 없는지 본 뒤 아래 본 학습을 돌립니다.

```bash
uv run python scripts/run_train.py --config configs/yolo_clspw.yaml
uv run python scripts/run_train.py --config configs/rcnn_balance.yaml
```

**켜졌는지 확인**: YOLO는 학습 시작 로그에 클래스 수만큼 숫자가 찍힙니다. 드문 클래스일수록 큽니다. 아래는 합성 데이터에서 확인한 출력입니다.

```text
Class weights: [1.29  0.912  0.89  0.89  0.99  0.912  0.936  1.178]
```

R-CNN은 로그가 따로 없으므로 `outputs/<실험>/config.yaml`에 `balance_power: 0.5`가 들어갔는지 봅니다.

**값 고르기**: 0(끔) → 0.5 → 1.0(빈도의 역수 그대로) 순서로 올려 봅니다. 너무 크면 흔한 알약 점수가 떨어져 평균이 오히려 내려갈 수 있습니다. 드문 클래스가 실제로 좋아졌는지는 `evaluate.py --exp <실험> --class-metrics`의 클래스별 AP로 확인합니다.

**데이터 쪽 D1과 동시에 켜지 않기**: 데이터 담당의 D1(`repeat_threshold`, 이미지 반복)과 T2는 같은 문제를 푸는 두 방법입니다. 둘을 동시에 켜면 드문 클래스 비중이 과하게 커지므로, 각각 baseline과 비교한 뒤 더 좋은 쪽 하나를 고릅니다.

## M1·M2·M3·T1. 모델과 학습 설정 바꾸기

모두 코드 수정 없이 configs/에 설정 파일을 새로 만들어 합니다(모델 A, 만드는 방법은 T2의 ①과 같음). 테스트 실행(`--epochs 1 --exp-name test_…`)으로 오류가 없는지 본 뒤 본 학습을 돌립니다.

| ID | 설정 파일 | 바꾸는 값 | 기대 | 주의 |
| --- | --- | --- | --- | --- |
| M1 | `yolo_v8m.yaml` | `weights: yolov8m.pt`, batch 8 | 더 큰 모델로 비슷한 알약 구별 | 학습 시간 약 2배, 232장에서는 과적합 가능 |
| M2 | `yolo11s.yaml` | `weights: yolo11s.pt` | 같은 크기에서 더 높은 정확도(COCO 기준) | 처음 실행 때 가중치 자동 다운로드 |
| M2 | `yolo26s.yaml` | `weights: yolo26s.pt` | 2026년 1월 버전, NMS가 필요 없는 구조 | P1의 `--nms-iou` 효과가 v8과 다를 수 있음 |
| M3 | `yolo_box_gain.yaml` | `yolo_args: {box: 10.0}` (기본 7.5) | 박스 위치를 더 정확하게 학습 → IoU 0.9 이상에서 점수 | 분류가 약해지면 오답 증가, 12 이상은 비추 |
| T1 | `yolo_long.yaml` | 120 epoch, `optimizer: AdamW`, `lr: 0.001`, `cos_lr`, `patience: 30` | baseline 50 epoch이 덜 학습됐을 때 | `optimizer: auto`면 `lr`이 무시되므로 함께 적음 |

아래 다섯 파일을 만듭니다. 모두 `yolo_baseline.yaml`을 물려받고 바뀐 값만 적습니다. 한 번에 다 만들지 않고, 돌릴 실험의 파일만 그때그때 만들어도 됩니다.

```yaml
# configs/yolo_v8m.yaml
# 실험 M1 (모델): 모델 크기 s → m
base: yolo_baseline.yaml
exp_name: 1027_modelA_yolov8m_v0_size
owner: modelA
note: 모델 크기 small → medium (M1)
model:
  weights: yolov8m.pt
train:
  batch_size: 8
```

```yaml
# configs/yolo11s.yaml
# 실험 M2 (모델): 최신 버전 YOLO11 (같은 크기 s)
base: yolo_baseline.yaml
exp_name: 1027_modelA_yolo11s_v0_version
owner: modelA
note: YOLOv8s → YOLO11s (M2)
model:
  weights: yolo11s.pt
```

```yaml
# configs/yolo26s.yaml
# 실험 M2 (모델): 최신 버전 YOLO26 (2026.01 공개, NMS가 필요 없는 구조)
base: yolo_baseline.yaml
exp_name: 1027_modelA_yolo26s_v0_version
owner: modelA
note: YOLOv8s → YOLO26s (M2)
model:
  weights: yolo26s.pt
```

```yaml
# configs/yolo_box_gain.yaml
# 실험 M3 (모델): 박스 위치 손실 비중 증가 — 지표가 IoU 0.75~0.95라 위치 정확도가 중요
base: yolo_baseline.yaml
exp_name: 1027_modelA_yolov8s_v0_box10
owner: modelA
note: box 손실 가중치 7.5 → 10 (M3)
yolo_args:
  box: 10.0
```

```yaml
# configs/yolo_long.yaml
# 실험 T1 (모델): 더 오래, 학습률을 직접 정해 학습
# optimizer가 auto면 lr이 무시되므로 AdamW를 명시한다.
base: yolo_baseline.yaml
exp_name: 1028_modelA_yolov8s_v0_long
owner: modelA
note: 120 epoch, AdamW lr 0.001, 코사인 스케줄, 조기 종료 30 (T1)
train:
  epochs: 120
  optimizer: AdamW
  lr: 0.001
yolo_args:
  cos_lr: true
  patience: 30
  close_mosaic: 15
```

```bash
# 예: M2 테스트 실행 → 본 학습
uv run python scripts/run_train.py --config configs/yolo11s.yaml --epochs 1 --exp-name test_m2
uv run python scripts/run_train.py --config configs/yolo11s.yaml
```

**실행 확인**: `cls_pw`, `box`, `cos_lr` 옵션은 합성 데이터로 1 epoch씩 실행해 오류 없이 학습·채점되는 것을 확인했습니다. 이 문서의 설정 파일에 쓴 `yolo_args` 키도 모두 설치된 Ultralytics(8.4.183)가 알아듣는 이름입니다.

**어느 것부터**: M2(버전) → M1(크기) → M3 순서를 권장합니다. 가장 좋은 버전·크기가 정해지면 그 위에 M3·T1을 올립니다. 다음 실험은 `base:`를 이긴 설정 파일(예: `yolo11s.yaml`)로 바꾸면 그 위에 쌓여집니다.

```yaml
# configs/yolo11s_box10.yaml — 이긴 실험 위에 쌓기
base: yolo11s.yaml
exp_name: 1029_modelA_yolo11s_v0_box10
owner: modelA
note: YOLO11s + box 10 (M2+M3)
yolo_args:
  box: 10.0
```

**R-CNN 쪽**: 백본 교체는 `rcnn.py`의 `BACKBONES` 표에 항목을 추가해야 하고, 학습 시간이 길어 우선순위가 낮습니다. R-CNN은 T2(`balance_power`)와 데이터 쪽 증강(`rcnn_aug.yaml`)에 집중하고, 주된 역할은 P2 앙상블의 두 번째 모델로 두는 것을 권장합니다.

## R1. R-CNN: 가장 좋았던 epoch 저장하기 (모델 B)

**문제**: baseline R-CNN은 마지막 epoch을 `best.pt`로 저장합니다. 끝에 가서 과적합되면 중간의 더 좋은 모델을 버리게 됩니다. YOLO는 Ultralytics가 알아서 최고 epoch을 고르므로 R-CNN만 고칩니다. 설정 `train.select_best: true`일 때만 켜지고, 기본은 꺼져 있어 baseline 결과는 그대로입니다.

모델 가이드(v1) 4장 예 1에서 이미 같은 코드를 넣었다면(`src/models/rcnn.py`에서 `Ctrl+F`로 `select_best`가 찾아지면) ①\~④은 건너뛰고 ⑤부터 합니다.

**① import 두 줄** (`src/models/rcnn.py` 맨 위). `from src.models.base import BaseDetector` 바로 위에 한 줄을 넣고, `from src.utils import ...` 줄에 `resolve_path`를 더합니다. 결과는 아래와 같아야 합니다.

```python
from src.interfaces import MAX_PILLS_PER_IMAGE, Prediction
from src.metrics import compute_map
from src.models.base import BaseDetector
from src.transforms import get_transforms
from src.utils import get_device, image_id_from_name, load_class_map, resolve_path, save_json
```

**② 검증 데이터 함수 추가**. `def fit(self, out_dir: Path) -> Path:` 줄을 찾아 그 **바로 위**에 아래를 붙여넣습니다(앞 공백 4칸, 클래스 안).

```python
    def _val_data(self):
        d = self.cfg["data"]
        records, _ = load_records(d["train_annotations"])
        val = load_split(records, f"{d['split_dir']}/{d['version']}/val.txt")
        ds = PillDataset(val, d["train_images"], load_class_map(d["class_map"]))
        paths = [resolve_path(d["train_images"]) / r["file_name"] for r in val]
        return paths, [ds[i][1] for i in range(len(ds))]

```

**③ 학습 루프 앞부분**. `fit` 안에서 아래 두 줄을 찾아

```python
        history = []
        for epoch in range(1, t["epochs"] + 1):
```

아래로 바꿉니다. 학습 루프 안에 `targets`라는 변수가 이미 있어서, 검증용은 `val_targets`라는 다른 이름을 씁니다.

```python
        history = []
        # 실험 R1: epoch마다 검증 mAP를 재서 가장 좋은 epoch을 best.pt로 (train.select_best)
        select_best = bool(t.get("select_best", False))
        if select_best:
            val_paths, val_targets = self._val_data()  # 학습 루프의 targets와 이름이 겹치지 않게
        best_score = -1.0
        best = out_dir / "best.pt"
        for epoch in range(1, t["epochs"] + 1):
```

**④ 학습 루프 끝부분**. 아래 부분을 찾아

```python
            print(f"[rcnn] epoch {epoch}: train_loss={avg:.4f}")

        # 스켈레톤은 마지막 epoch을 best로 저장. 고도화: epoch마다 검증 mAP로 best 선택
        best = out_dir / "best.pt"
        torch.save(self.model.state_dict(), best)
```

아래로 바꿉니다. `if select_best:`가 `print`와 같은 깊이(공백 12칸)여야 epoch마다 실행됩니다.

```python
            print(f"[rcnn] epoch {epoch}: train_loss={avg:.4f}")
            if select_best:
                score = compute_map(
                    self.predict(val_paths), val_targets, self.cfg["eval"].get("iou_thresholds")
                )["map"]
                history[-1]["val_map"] = score
                print(f"[rcnn] epoch {epoch}: val_map={score:.4f}")
                if score > best_score:
                    best_score = score
                    torch.save(self.model.state_dict(), best)

        if not select_best:  # 기본(baseline): 마지막 epoch을 best로 저장
            torch.save(self.model.state_dict(), best)
```

**⑤ 설정 파일 만들기**: `configs/rcnn_select_best.yaml`

```yaml
# configs/rcnn_select_best.yaml
# 실험 R1 (R-CNN): epoch마다 검증해 가장 좋았던 epoch을 best.pt로 저장 (학습이 조금 느려짐)
base: rcnn_baseline.yaml
exp_name: 1027_modelB_frcnn_v0_selectbest
owner: modelB
note: epoch별 검증 mAP로 최고 epoch 선택 (R1)
train:
  select_best: true
```

**⑥ 확인**: 2 epoch만 돌려 epoch마다 `val_map` 줄이 찍히는지 봅니다. 다른 R-CNN 실험(`rcnn_balance.yaml` 등)에도 `train:` 아래 `select_best: true`를 넣으면 같이 적용됩니다.

```bash
uv run python scripts/run_train.py --config configs/rcnn_select_best.yaml --epochs 2 --exp-name test_r1
#   [rcnn] epoch 1: train_loss=...
#   [rcnn] epoch 1: val_map=0.xxxx
#   [rcnn] epoch 2: train_loss=...
#   [rcnn] epoch 2: val_map=0.xxxx
uv run python scripts/run_train.py --config configs/rcnn_baseline.yaml --epochs 1 --exp-name test_r1_off
#   val_map 줄이 없으면 기본(꺼짐) 동작 정상
uv run ruff check . && uv run pytest -q
```

## P1·P4·P2. 재학습 없이 점수 올리기

**한 줄 요약**: 학습된 `best.pt`는 그대로 두고 **예측하는 방식만** 바꿉니다. 몇 분이면 끝나고, 합성 데이터에서는 아래처럼 점수가 크게 올랐습니다.

| 방법 | 명령 (기준: `sample_yolo_e30`) | mAP@\[0.75:0.95\] | mAP75 |
| --- | --- | --- | --- |
| 기본 예측 | `evaluate.py --exp sample_yolo_e30` | 0.214 | 0.465 |
| P4 TTA | `… --tta` | 0.259 | 0.506 |
| P2 앙상블 NMS (+ img416 모델) | `… --ensemble sample_yolo_img416 --method nms` | 0.238 | 0.539 |
| P2 앙상블 WBF (+ img416 모델) | `… --ensemble sample_yolo_img416 --method wbf` | **0.291** | **0.593** |

합성 데이터(검증 16장) 결과라 실제 데이터에서는 폭이 다릅니다. 다만 “박스를 버리는 NMS보다 박스를 평균 내는 WBF가 높은 IoU에서 유리하다”는 방향은 이번 지표와 잘 맞습니다.

P1은 baseline 코드로 바로 됩니다. P4·P2는 아래 **코드 준비 세 단계**(P4 → P2 → PS)가 main에 병합된 뒤에 `--tta`, `--ensemble`, `--method` 옵션이 생깁니다.

### 코드 준비 ① P4: YOLO 예측에 TTA 스위치 (모델 A, M0 병합 후)

`src/models/yolo.py`의 `predict` 함수에서 `self.model.predict(`를 찾고, 그 안의 `device=self._device_arg(),` 줄과 `verbose=False,` 줄 사이에 한 줄을 넣습니다.

```python
                device=self._device_arg(),
                augment=bool(ev.get("tta", False)),  # 실험 P4: TTA (느려짐)
                verbose=False,
```

설정의 `eval.tta`가 없으면 `False`라 baseline 예측은 그대로입니다. 확인: `uv run pytest -q`(4 passed), `uv run ruff check .`. 점수 비교는 PS가 끝난 뒤 `evaluate.py --tta`로 합니다.

### 코드 준비 ② P2: WBF 함수 추가 (모델 B)

`src/ensemble.py` 위쪽 import 줄을 바꿉니다.

```python
# 전
from torchvision.ops import batched_nms
# 후
from torchvision.ops import batched_nms, box_iou
```

그리고 파일 **맨 끝**(`ensemble_nms` 함수의 `return merged` 아래, 빈 줄 두 개 띄고)에 아래 함수를 통째로 붙여넣습니다.

```python
def ensemble_wbf(
    pred_lists: list[list[Prediction]],
    iou_thr: float = 0.55,
    max_det: int = MAX_PILLS_PER_IMAGE,
    weights: list[float] | None = None,
) -> list[Prediction]:
    """Weighted Boxes Fusion. 같은 클래스에서 IoU가 iou_thr 이상인 박스를 한 덩어리로 묶어
    좌표는 (점수 x 모델 가중치) 가중 평균, 점수는 평균 x (참여 모델 수 / 전체 모델 수).
    한 모델만 찾은 박스는 점수가 깎여 뒤로 밀린다.
    """
    weights = weights or [1.0] * len(pred_lists)
    n_models = len(pred_lists)
    by_file: dict[str, list[tuple[Prediction, float, int]]] = {}
    for k, (preds, w) in enumerate(zip(pred_lists, weights, strict=True)):
        for p in preds:
            by_file.setdefault(p["file_name"], []).append((p, w, k))

    merged: list[Prediction] = []
    for file_name, items in by_file.items():
        out_boxes, out_scores, out_labels = [], [], []
        labels_all = torch.cat([p["labels"] for p, _, _ in items]) if items else torch.empty(0)
        for label in labels_all.unique().tolist():
            entries = []  # (box, score, weight, model)
            for p, w, k in items:
                m = p["labels"] == label
                for b, s in zip(p["boxes"][m], p["scores"][m], strict=True):
                    entries.append((b.float(), float(s), w, k))
            entries.sort(key=lambda e: -e[1] * e[2])
            clusters: list[list[tuple]] = []
            fused: list[torch.Tensor] = []
            for e in entries:
                if fused:
                    ious = box_iou(e[0][None], torch.stack(fused))[0]
                    j = int(ious.argmax())
                    if ious[j] >= iou_thr:
                        clusters[j].append(e)
                        ws = torch.tensor([c[1] * c[2] for c in clusters[j]])
                        fused[j] = (torch.stack([c[0] for c in clusters[j]]) * ws[:, None]).sum(
                            0
                        ) / ws.sum()
                        continue
                clusters.append([e])
                fused.append(e[0])
            for box, cl in zip(fused, clusters, strict=True):
                score = sum(c[1] * c[2] for c in cl) / sum(c[2] for c in cl)
                n_used = len({c[3] for c in cl})
                out_boxes.append(box)
                out_scores.append(score * min(n_used, n_models) / n_models)
                out_labels.append(label)
        if out_boxes:
            boxes = torch.stack(out_boxes)
            scores = torch.tensor(out_scores, dtype=torch.float32)
            labels = torch.tensor(out_labels, dtype=torch.long)
            order = scores.argsort(descending=True)[:max_det]
            boxes, scores, labels = boxes[order], scores[order], labels[order]
        else:
            boxes = torch.zeros((0, 4))
            scores = torch.zeros(0)
            labels = torch.zeros(0, dtype=torch.long)
        merged.append(
            Prediction(
                image_id=items[0][0]["image_id"],
                file_name=file_name,
                boxes=boxes,
                scores=scores,
                labels=labels,
            )
        )
    return merged
```

**확인**: 두 모델이 거의 같은 자리를 찾은 가짜 예측으로 함수만 시험합니다. 두 박스의 평균 자리와 점수가 나오면 성공입니다.

```bash
uv run python -c "import torch; from src.ensemble import ensemble_wbf; a={'image_id':1,'file_name':'x.png','boxes':torch.tensor([[10.,10,50,50]]),'scores':torch.tensor([0.9]),'labels':torch.tensor([0])}; b={**a,'boxes':torch.tensor([[12.,10,52,50]]),'scores':torch.tensor([0.7])}; r=ensemble_wbf([[a],[b]])[0]; print(r['boxes'], r['scores'])"
# tensor([[10.8750, 10.0000, 50.8750, 50.0000]]) tensor([0.8000])
uv run ruff check . && uv run pytest -q
```

### 코드 준비 ③ PS: 평가·제출 스크립트에 옵션 연결 (Experimentation Lead, P2·P4 병합 후)

세 파일을 고칩니다. 두 스크립트는 바뀜 곳이 많아 **파일 전체를 바꿔 붙여넣는 방식**입니다(`Ctrl+A` → 붙여넣기 → 저장). 데이터 D5의 `--val-config` 옵션도 여기서 함께 들어갑니다.

**(1) `src/infer.py`**: `def predict_test(`로 시작하는 함수 전체(끝의 `return preds`까지)를 아래로 바꿉니다. TTA 예측은 파일 이름을 달리 해 기본 예측과 섞이지 않게 합니다.

```python
def test_pred_file(exp_name: str, tta: bool = False) -> Path:
    return OUTPUT_DIR / exp_name / ("preds_test_tta.json" if tta else "preds_test.json")


def predict_test(
    exp_name: str,
    image_dir: str | Path | None = None,
    conf_thr: float | None = None,
    tta: bool = False,
) -> list[Prediction]:
    cfg, model = load_experiment(exp_name)
    if tta:  # 실험 P4: YOLO 예측 TTA (model.cfg와 같은 dict)
        cfg["eval"]["tta"] = True
    image_dir = image_dir or cfg["data"]["test_images"]
    paths = list_images(image_dir)
    preds = model.predict(paths, conf_thr=conf_thr)
    out = test_pred_file(exp_name, tta)
    save_json(predictions_to_json(preds), out)
    print(f"[{exp_name}] 테스트 이미지 {len(paths)}장 예측 → outputs/{exp_name}/{out.name}")
    return preds
```

**(2) `scripts/evaluate.py` 전체**

```python
"""학습 없이 평가만 다시 하기 (모델 A·B, 실험 P1: 임계값·NMS 조정).

실행:
    uv run python scripts/evaluate.py --exp 1028_modelA_yolov8s_v0_baseline
    uv run python scripts/evaluate.py --exp 1028_modelA_yolov8s_v0_baseline --nms-iou 0.6 --max-det 4
    uv run python scripts/evaluate.py --exp <실험> --class-metrics      # 클래스별 AP까지
    uv run python scripts/evaluate.py --exp <YOLO 실험> --tta           # 예측 TTA (실험 P4)
    uv run python scripts/evaluate.py --exp <실험A> --ensemble <실험B> --method wbf   # 앙상블 검증 (P2)
    uv run python scripts/evaluate.py --exp <baseline> --val-config configs/yolo_v1_clean.yaml
        # 다른 데이터 버전(v1 정제)의 검증 정답으로 채점: 정제 전·후 모델을 같은 기준으로 비교 (D5)

결과: 화면 출력 + outputs/{exp}/eval_{설정}.json
학습된 가중치(best.pt)는 그대로 두고, 예측 설정만 바꿔 검증 점수를 다시 계산한다.
좋은 값을 찾으면 설정 파일의 eval 블록에 반영해 다음 학습·제출에 쓴다.
"""

from __future__ import annotations

import argparse

from src.ensemble import ensemble_nms, ensemble_wbf
from src.infer import load_experiment
from src.interfaces import OUTPUT_DIR
from src.metrics import compute_map
from src.train import val_targets
from src.utils import load_config, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--conf", type=float, default=None, help="eval.conf_thr 대신 쓸 값")
    ap.add_argument("--nms-iou", type=float, default=None, help="eval.nms_iou 대신 쓸 값 (YOLO)")
    ap.add_argument("--max-det", type=int, default=None, help="eval.max_det 대신 쓸 값")
    ap.add_argument("--class-metrics", action="store_true", help="클래스별 AP 계산 (느림)")
    ap.add_argument("--tta", action="store_true", help="YOLO 예측 TTA (뒤집기·크기 변경 합침)")
    ap.add_argument("--ensemble", nargs="+", default=None, help="함께 합칠 다른 실험 이름")
    ap.add_argument("--method", choices=["nms", "wbf"], default="wbf", help="앙상블 방식")
    ap.add_argument("--val-config", default=None, help="이 설정의 data 블록(검증셋·정답)으로 채점")
    args = ap.parse_args()

    cfg, model = load_experiment(args.exp)
    ev = cfg["eval"]
    for key, value in (
        ("conf_thr", args.conf),
        ("nms_iou", args.nms_iou),
        ("max_det", args.max_det),
    ):
        if value is not None:
            ev[key] = value
    if args.tta:
        ev["tta"] = True
    if hasattr(model, "model") and hasattr(model.model, "roi_heads"):  # Faster R-CNN
        model.model.roi_heads.detections_per_img = ev["max_det"]

    data_cfg = cfg
    if args.val_config:  # 모델은 그대로, 채점할 검증 데이터만 바꾼다
        data_cfg = {**cfg, "data": {**cfg["data"], **load_config(args.val_config)["data"]}}
    paths, targets = val_targets(data_cfg)
    preds = model.predict(paths)
    if args.ensemble:
        pred_lists = [preds]
        for other in args.ensemble:
            _, m = load_experiment(other)
            pred_lists.append(m.predict(paths))
        merge = ensemble_wbf if args.method == "wbf" else ensemble_nms
        preds = merge(pred_lists, max_det=ev["max_det"])
    metrics = compute_map(
        preds, targets, ev.get("iou_thresholds"), class_metrics=args.class_metrics
    )

    tag = f"conf{ev['conf_thr']}_nms{ev.get('nms_iou')}_det{ev['max_det']}"
    if args.tta:
        tag += "_tta"
    if args.ensemble:
        tag += f"_{args.method}_" + "+".join(args.ensemble)
    if args.val_config:
        tag += "_val" + data_cfg["data"]["version"]
    save_json(
        {"exp_name": args.exp, "eval": ev, **metrics}, OUTPUT_DIR / args.exp / f"eval_{tag}.json"
    )
    print(f"[{args.exp}] {tag}")
    print(f"  mAP@[0.75:0.95] = {metrics.get('map')}   mAP75 = {metrics.get('map_75')}")
    print(f"  저장: outputs/{args.exp}/eval_{tag}.json")


if __name__ == "__main__":
    main()
```

**(3) `scripts/make_submission.py` 전체**

```python
"""Kaggle 제출 파일 생성 (Experimentation Lead).

실행:
    uv run python scripts/make_submission.py --exp 1028_modelA_yolov8s_v0_baseline
    uv run python scripts/make_submission.py --exp expA expB          # 두 모델 앙상블 (NMS)
    uv run python scripts/make_submission.py --exp expA expB --method wbf --weights 2 1   # WBF (실험 P2)
    uv run python scripts/make_submission.py --exp expA --tta                              # TTA (실험 P4)

결과: submissions/{exp}.csv

제출 형식은 대회 sample_submission.csv와 반드시 비교해서 맞춘다.
- 박스: 팀 표준 xyxy → 대회 형식 (기본: COCO xywh)
- 클래스: label_idx → 대회 원본 category_id
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.ensemble import ensemble_nms, ensemble_wbf
from src.infer import predict_test, test_pred_file
from src.interfaces import OUTPUT_DIR, PROJECT_ROOT
from src.utils import (
    label_to_category,
    load_class_map,
    load_config,
    load_json,
    predictions_from_json,
    xyxy_to_xywh,
)

COLUMNS = [
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "score",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True, help="실험 이름 (여러 개면 앙상블)")
    ap.add_argument("--reuse", action="store_true", help="이미 있는 preds_test.json 재사용")
    ap.add_argument("--method", choices=["nms", "wbf"], default="nms", help="앙상블 방식")
    ap.add_argument("--weights", type=float, nargs="+", default=None, help="모델별 가중치")
    ap.add_argument("--iou", type=float, default=None, help="앙상블 IoU (기본 nms 0.5, wbf 0.55)")
    ap.add_argument("--tta", action="store_true", help="YOLO 예측 TTA (실험 P4)")
    args = ap.parse_args()

    pred_lists = []
    for exp in args.exp:
        path = test_pred_file(exp, args.tta)
        if args.reuse and path.exists():
            pred_lists.append(predictions_from_json(load_json(path)))
        else:
            pred_lists.append(predict_test(exp, tta=args.tta))
    if len(pred_lists) == 1:
        preds = pred_lists[0]
    elif args.method == "wbf":
        preds = ensemble_wbf(pred_lists, iou_thr=args.iou or 0.55, weights=args.weights)
    else:
        preds = ensemble_nms(pred_lists, iou_thr=args.iou or 0.5, weights=args.weights)

    cfg = load_config(OUTPUT_DIR / args.exp[0] / "config.yaml")
    l2c = label_to_category(load_class_map(cfg["data"]["class_map"]))
    conf_thr = cfg.get("submission", {}).get("conf_thr", 0.0)

    rows = []
    for p in sorted(preds, key=lambda x: x["image_id"]):
        keep = p["scores"] >= conf_thr
        for box, score, label in zip(
            xyxy_to_xywh(p["boxes"][keep]), p["scores"][keep], p["labels"][keep], strict=True
        ):
            x, y, w, h = (round(float(v), 2) for v in box)
            rows.append(
                {
                    "annotation_id": len(rows) + 1,
                    "image_id": p["image_id"],
                    "category_id": l2c[int(label)],
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "score": round(float(score), 5),
                }
            )
    df = pd.DataFrame(rows, columns=COLUMNS)

    suffix = (f"_{args.method}" if len(args.exp) > 1 else "") + ("_tta" if args.tta else "")
    out = PROJECT_ROOT / "submissions" / ("+".join(args.exp) + suffix + ".csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"제출 파일: {out} ({len(df)}행, 이미지 {df['image_id'].nunique()}장)")
    print("Kaggle 제출 전 sample_submission.csv와 컨럼 이름·순서가 같은지 확인하세요.")


if __name__ == "__main__":
    main()
```

**확인**: 학습된 YOLO 실험 두 개(예: baseline과 v1에서 만든 img1024)로 아래를 돌려 점수 줄이 매번 찍히고, 기본 명령의 점수가 고치기 전과 같으면 성공입니다.

```bash
uv run python scripts/evaluate.py --exp <실험A>                             # 고치기 전과 같은 점수여야 함
uv run python scripts/evaluate.py --exp <실험A> --tta
uv run python scripts/evaluate.py --exp <실험A> --ensemble <실험B> --method wbf
uv run python scripts/make_submission.py --exp <실험A> <실험B> --method wbf --weights 2 1
#   제출 파일: .../submissions/<실험A>+<실험B>_wbf.csv (...행, 이미지 842장)
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

### P1. 확신도·NMS 설정 찾기

```bash
uv run python scripts/evaluate.py --exp <실험> --nms-iou 0.6
uv run python scripts/evaluate.py --exp <실험> --nms-iou 0.7
```

알약끼리 거의 겹치지 않으므로 NMS IoU를 0.5\~0.7 사이에서 찾습니다. `conf_thr`는 mAP 계산상 낮을수록 유리해서 기본 0.001을 유지합니다. 좋은 값은 제출 전에 `outputs/<실험>/config.yaml`의 `eval.nms_iou`에 넣습니다(제출은 그 파일을 읽음).

### P4. 예측 TTA (YOLO)

예측할 때 이미지를 뒤집고 크기를 바꿔 여러 번 예측한 뒤 합칩니다. 예측 시간이 약 3배이지만 제출에는 문제가 없습니다.

```bash
uv run python scripts/evaluate.py      --exp <YOLO 실험> --tta     # 검증 점수 비교
uv run python scripts/make_submission.py --exp <YOLO 실험> --tta     # → submissions/<실험>_tta.csv
```

R-CNN에는 TTA 옵션이 없어 `--tta`를 줘도 아무 변화가 없습니다.

### P2. 두 모델 합치기 (WBF)

서로 다른 모델(예: YOLO + R-CNN, 또는 설정이 다른 YOLO 두 개)의 예측을 합칩니다. 같은 알약을 가리키는 박스들의 좌표를 점수로 가중 평균하고, 한 모델만 찾은 박스는 점수를 깎습니다.

```bash
# ① 검증으로 효과 확인 (첫 실험이 기준, --ensemble 뒤가 함께 합칠 모델)
uv run python scripts/evaluate.py --exp <실험A> --ensemble <실험B> --method wbf
uv run python scripts/evaluate.py --exp <실험A> --ensemble <실험B> --method nms   # 비교용
# ② 제출 파일 (더 좋은 모델에 가중치를 더 줄 수 있음)
uv run python scripts/make_submission.py --exp <실험A> <실험B> --method wbf --weights 2 1
# → submissions/<실험A>+<실험B>_wbf.csv
```

| 옵션 | 기본 | 뜻 |
| --- | --- | --- |
| `--method` | evaluate: wbf / submission: nms | 합치는 방식 |
| `--weights` | 모두 1 | 모델별 신뢰도 (점수가 높은 모델을 2) |
| `--iou` | wbf 0.55 / nms 0.5 | 같은 알약으로 볼 겹침 정도 |

**어떤 모델끼리 합치나**: 점수가 비슷하고 틀리는 이미지가 서로 다른 모델끼리가 가장 효과가 큽니다(YOLO + R-CNN, 해상도가 다른 YOLO 두 개). 점수 차이가 크면 약한 모델이 오히려 끌어내릴 수 있으므로 ①의 검증 점수로 판단합니다. TTA와 함께 쓰려면 제출 명령에 `--tta`를 더합니다.

### 마지막: 테스트 추가 (Experimentation Lead, 선택)

데이터 D1·D3과 모델 P2가 모두 main에 병합된 뒤, 새 기능이 나중에 깨지지 않게 자동 검사 3개를 `tests/test_pipeline.py` **맨 끝**에 붙여넣습니다. 붙인 뒤 `uv run pytest -q` 결과가 **7 passed**면 성공입니다.

```python
# ── 개선 실험 기능 (v2 가이드) ────────────────────────────────


def test_rotate90_keeps_boxes_tight():
    """D3: 90도 회전 후에도 박스가 알약 픽셀을 정확히 감싼다."""
    import random

    import numpy as np

    from src.transforms import Rotate90

    img = np.zeros((100, 200, 3), np.uint8)
    img[10:30, 40:90] = 255
    boxes = np.array([[40, 10, 90, 30]], np.float32)
    for seed in range(8):
        random.seed(seed)
        out, b, _ = Rotate90(1.0)(img, boxes, np.array([0]))
        ys, xs = np.where(out[..., 0] > 0)
        assert b[0].tolist() == [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


def test_oversample_repeats_rare_images(sample_records):
    """D1: 드문 클래스 이미지는 늘고, threshold 0이면 그대로다."""
    from src.dataset import oversample

    records, _ = sample_records
    assert len(oversample(records, 0.0)) == len(records)
    assert len(oversample(records, 0.9)) > len(records)
    assert oversample(records, 0.9, seed=1) == oversample(records, 0.9, seed=1)


def test_wbf_averages_overlapping_boxes():
    """P2: 두 모델이 겹치게 찾은 박스는 점수 가중 평균, 한 모델만 찾은 박스는 점수 절반."""
    from src.ensemble import ensemble_wbf
    from src.interfaces import Prediction

    a = Prediction(
        image_id=1,
        file_name="x.png",
        boxes=torch.tensor([[10.0, 10, 50, 50], [100, 100, 140, 140]]),
        scores=torch.tensor([0.9, 0.8]),
        labels=torch.tensor([0, 1]),
    )
    b = Prediction(
        image_id=1,
        file_name="x.png",
        boxes=torch.tensor([[12.0, 10, 52, 50]]),
        scores=torch.tensor([0.7]),
        labels=torch.tensor([0]),
    )
    out = ensemble_wbf([[a], [b]])[0]
    assert out["boxes"][0, 0].item() == pytest.approx((10 * 0.9 + 12 * 0.7) / 1.6)
    assert out["scores"].tolist() == pytest.approx([0.8, 0.4])
```

## 추가 실험 (시간이 더 들어도 해 볼 것)

위 실험을 끝낸 뒤 남은 시간에 합니다. 모두 합성 연습 데이터로 끝까지 돌아가는 것을 확인했고, 기본값은 모두 꺼져 있어 기존 결과는 바뀌지 않습니다.

| 순서 | ID | 실험 | 담당 | 만드는 것 | 비용 |
| --- | --- | --- | --- | --- | --- |
| 1 | X0 | 명령줄에서 설정 값 바꾸기 `--set` | Experimentation Lead | `src/utils.py`, `run_train.py`, `export_yolo.py` | 10분 |
| 2 | M4 | 탐지 + 분류 2단계 | 모델 B (+EL 등록) | 새 `src/models/two_stage.py`, 등록 2곳, 새 설정 | 하루 |
| 2 | M5 | 큰 모델 + 해상도 1024 | 모델 A | 새 설정 | 학습 4\~6배 |
| 2 | T3 | 학습 설정 자동 탐색 | 모델 A | 새 `scripts/tune_yolo.py` | 반나절\~하루 (GPU) |
| 3 | T4 | 시드·fold 앙상블 | 모델 A·B | 없음 (X0 + 데이터 D7 필요) | 학습 3\~5배 |
| 4 | P5 | 최종 제출: 전체 데이터 학습 + 앙상블 | Experimentation Lead | 새 설정 `yolo_full.yaml` | 학습 1\~5번 |

### X0. `--set` 옵션 (Experimentation Lead, 가장 먼저)

설정 파일을 새로 만들지 않고 값 몇 개만 바꿔 여러 번 학습하기 위한 기능입니다(예: `--set data.version=v0_f1 seed=7`). D7·T4가 이것을 씁니다. 바꾼 값은 `outputs/<실험>/config.yaml`에 그대로 남아 평가·제출에 문제가 없습니다.

**① `src/utils.py`**: `def save_yaml(` 줄 바로 위에 함수를 추가합니다.

```python
def apply_overrides(cfg: dict, items: list[str] | None) -> dict:
    """명령줄의 --set data.version=v0_f1 seed=7 처럼 설정 값 몇 개를 바꾼다 (실험 D7·T4).

    값은 YAML로 읽으므로 숫자·true/false·리스트도 그대로 쓸 수 있다.
    """
    for item in items or []:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise ValueError(f"--set 형식은 키=값 입니다: {item}")
        node = cfg
        *parents, last = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[last] = yaml.safe_load(raw)
    return cfg


```

**② `scripts/run_train.py`** 세 곳

```python
# (가) import 줄
from src.utils import apply_overrides, load_config

# (나) --exp-name 줄 바로 아래에 추가
    ap.add_argument(
        "--set", nargs="*", default=None, help="설정 값 바꾸기 (예: data.version=v0_f1 seed=7)"
    )

# (다) cfg = load_config(args.config) 줄을 아래로
    cfg = apply_overrides(load_config(args.config), args.set)
```

**③ `scripts/export_yolo.py`** 세 곳

```python
# (가) from src.utils import ( 아래 첫 줄에 추가
    apply_overrides,

# (나) --config 줄 바로 아래에 추가
    ap.add_argument(
        "--set", nargs="*", default=None, help="설정 값 바꾸기 (예: data.version=v0_f1)"
    )

# (다) cfg = load_config(args.config) 줄을 아래로
    cfg = apply_overrides(load_config(args.config), args.set)
```

**④ 확인**: 1 epoch만 돌리고 저장된 설정에 바꾼 값이 들어갔는지 봅니다.

```bash
uv run python scripts/run_train.py --config configs/yolo_baseline.yaml --epochs 1 --exp-name test_x0 --set seed=7
# outputs/test_x0/config.yaml 에 seed: 7
uv run ruff check . && uv run pytest -q
```

### M4. 탐지 + 분류 2단계 (모델 B)

**문제**: 하나의 탐지 모델이 위치 찾기와 수십 종 구별을 한꺼번에 하면, 각인·글자처럼 작은 차이로 구별하는 알약에서 틀리기 쉽습니다. 04 오분류 분석에서 “위치는 맞는데 이름이 틀린” 경우가 많으면 시도합니다.

**방식**: 이미 학습이 끝난 탐지 실험(예: YOLO 최고 모델)이 박스를 찾고, 정답 박스로 잘라낸 알약 사진으로 학습한 분류 모델(torchvision ResNet18 등)이 박스마다 이름을 다시 고릅니다. 박스 위치는 바꾸지 않고 이름과 점수만 바뀝니다. 평가·제출·데모는 다른 모델과 똑같이 실험 이름으로 씁니다.

**① 새 파일 `src/models/two_stage.py`** (전체를 그대로 복사)

```python
"""탐지 + 분류 2단계 모델 (실험 M4, 모델 B).

1단계: 이미 학습이 끝난 탐지 실험(model.detector_exp)이 박스를 찾는다.
2단계: 정답 박스로 잘라낸 알약 사진으로 학습한 분류 모델이 박스마다 이름을 다시 고른다.
       작은 알약을 크게 키워 보므로 각인·글자처럼 작은 차이를 더 잘 본다.
최종: 확률 = cls_weight x 분류 확률 + (1 - cls_weight) x 탐지 결과(원-핫)
      이름 = 확률이 가장 큰 클래스, 점수 = 탐지 점수 x 그 확률

설정 예: configs/two_stage.yaml. 평가·제출·데모는 다른 모델과 똑같이 실험 이름으로 쓴다.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.nn import functional as F  # noqa: N812

from src.dataset import load_records, load_split, read_image_rgb
from src.interfaces import OUTPUT_DIR, Prediction
from src.models.base import BaseDetector
from src.utils import category_to_label, get_device, load_class_map, resolve_path

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_classifier(name: str, num_classes: int, pretrained: bool) -> torch.nn.Module:
    """torchvision 분류 모델의 마지막 층을 클래스 수에 맞게 바꾼다 (resnet18, efficientnet_b0 등)."""
    model = torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)
    if hasattr(model, "fc"):  # resnet 계열
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:  # efficientnet, convnext, mobilenet 계열
        last = model.classifier[-1]
        model.classifier[-1] = torch.nn.Linear(last.in_features, num_classes)
    return model


def crop(img: np.ndarray, box, size: int, pad: float, jitter: float = 0.0) -> np.ndarray:
    """박스를 pad 비율만큼 넓혀 잘라 size x size로. jitter는 학습 때 박스를 조금 흔든다."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    if jitter:
        x1 += random.uniform(-jitter, jitter) * bw
        x2 += random.uniform(-jitter, jitter) * bw
        y1 += random.uniform(-jitter, jitter) * bh
        y2 += random.uniform(-jitter, jitter) * bh
    x1, x2 = max(0, int(x1 - pad * bw)), min(w, int(x2 + pad * bw))
    y1, y2 = max(0, int(y1 - pad * bh)), min(h, int(y2 + pad * bh))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(img[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


def to_tensor(crops: list[np.ndarray]) -> torch.Tensor:
    x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN) / STD


def augment(c: np.ndarray) -> np.ndarray:
    c = np.rot90(c, random.randint(0, 3))
    if random.random() < 0.5:
        c = c[:, ::-1]
    c = np.clip(c.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(c)


class TwoStageDetector(BaseDetector):
    name = "two_stage"

    def __init__(self, cfg: dict, num_classes: int):
        super().__init__(cfg, num_classes)
        m = cfg["model"]
        self.size = int(m.get("crop_size", 224))
        self.pad = float(m.get("crop_pad", 0.1))
        self.device = get_device(cfg["train"].get("device"))
        self.classifier = None
        self.detector = None

    def _crops(self, split: str) -> tuple[list[np.ndarray], list[int], list]:
        d = self.cfg["data"]
        records, _ = load_records(d["train_annotations"])
        recs = load_split(records, f"{d['split_dir']}/{d['version']}/{split}.txt")
        cat2label = category_to_label(load_class_map(d["class_map"]))
        image_dir = resolve_path(d["train_images"])
        images, labels, boxes = [], [], []
        for r in recs:
            img = read_image_rgb(image_dir / r["file_name"])
            for box, cid in zip(r["boxes"], r["category_ids"], strict=True):
                images.append(img)
                boxes.append(box)
                labels.append(cat2label[cid])
        return images, labels, boxes

    def fit(self, out_dir: Path) -> Path:
        m, t = self.cfg["model"], self.cfg["train"]
        self.classifier = build_classifier(
            m.get("classifier", "resnet18"), self.num_classes, bool(m.get("pretrained", True))
        ).to(self.device)
        tr_img, tr_lbl, tr_box = self._crops("train")
        va_img, va_lbl, va_box = self._crops("val")
        va_crops = [crop(i, b, self.size, self.pad) for i, b in zip(va_img, va_box, strict=True)]
        va_x = to_tensor(va_crops) if va_crops else None
        opt = torch.optim.AdamW(self.classifier.parameters(), lr=float(t.get("lr", 3e-4)))
        bs = int(t.get("batch_size", 32))
        best_acc, best = -1.0, out_dir / "best.pt"
        print(
            f"[two_stage] 학습 알약 {len(tr_lbl)}개, 검증 알약 {len(va_lbl)}개, 클래스 {self.num_classes}개"
        )
        for epoch in range(1, int(t["epochs"]) + 1):
            self.classifier.train()
            order = list(range(len(tr_lbl)))
            random.shuffle(order)
            total = 0.0
            for s in range(0, len(order), bs):
                idx = order[s : s + bs]
                x = to_tensor(
                    [augment(crop(tr_img[i], tr_box[i], self.size, self.pad, 0.05)) for i in idx]
                ).to(self.device)
                y = torch.tensor([tr_lbl[i] for i in idx], device=self.device)
                loss = F.cross_entropy(self.classifier(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            acc = self._accuracy(va_x, va_lbl) if va_x is not None else 0.0
            print(
                f"[two_stage] epoch {epoch}: loss={total / max(1, len(order)):.4f} val_acc={acc:.3f}"
            )
            if acc > best_acc:
                best_acc = acc
                torch.save(self.classifier.state_dict(), best)
        # best.pt에는 분류 모델만 저장한다. 어떤 탐지 실험과 묶였는지는 설정 사본으로 남긴다
        detector_cfg = OUTPUT_DIR / m["detector_exp"] / "config.yaml"
        shutil.copy(detector_cfg, out_dir / "detector_config.yaml")
        return best

    @torch.no_grad()
    def _accuracy(self, x: torch.Tensor, labels: list[int]) -> float:
        self.classifier.eval()
        pred = torch.cat(
            [
                self.classifier(x[s : s + 64].to(self.device)).argmax(1).cpu()
                for s in range(0, len(x), 64)
            ]
        )
        return float((pred == torch.tensor(labels)).float().mean())

    def load(self, weights: str | Path) -> None:
        from src.infer import load_experiment  # 실험 폴더에서 1단계 탐지 모델을 읽는다

        m = self.cfg["model"]
        self.classifier = build_classifier(m.get("classifier", "resnet18"), self.num_classes, False)
        self.classifier.load_state_dict(torch.load(weights, map_location="cpu"))
        self.classifier.to(self.device).eval()
        _, self.detector = load_experiment(m["detector_exp"])
        if self.detector.num_classes != self.num_classes:
            raise ValueError(
                "탐지 실험과 클래스 수가 다릅니다. 같은 class_map으로 학습한 탐지 실험을 쓰세요."
            )

    @torch.no_grad()
    def predict(self, image_paths: list[Path], conf_thr: float | None = None) -> list[Prediction]:
        self.detector.cfg["eval"].update(self.cfg["eval"])  # --tta, --nms-iou 등을 1단계에도 적용
        w = float(self.cfg["model"].get("cls_weight", 0.5))
        out = []
        for path, p in zip(image_paths, self.detector.predict(image_paths, conf_thr), strict=True):
            if len(p["boxes"]) == 0:
                out.append(p)
                continue
            img = read_image_rgb(path)
            x = to_tensor([crop(img, b, self.size, self.pad) for b in p["boxes"].tolist()])
            probs = self.classifier(x.to(self.device)).softmax(1).cpu()
            mixed = w * probs + (1 - w) * F.one_hot(p["labels"], self.num_classes).float()
            conf, labels = mixed.max(1)
            out.append(
                Prediction(
                    image_id=p["image_id"],
                    file_name=p["file_name"],
                    boxes=p["boxes"],
                    scores=p["scores"] * conf,
                    labels=labels.long(),
                )
            )
        return out
```

**② 모델 등록** `src/models/__init__.py` (Experimentation Lead 승인): 마지막 `raise ValueError(...)` 줄을 아래로 바꿉니다.

```python
    if family == "two_stage":  # 실험 M4: 탐지 + 분류 2단계
        from src.models.two_stage import TwoStageDetector

        return TwoStageDetector(cfg, num_classes)
    raise ValueError(f"알 수 없는 model.family: {family} (yolo / rcnn / two_stage)")
```

**③ 실험 기록표 칸** `src/train.py` (Experimentation Lead): 기록표의 model 칸이 새 모델에서 오류가 나지 않게 합니다. `def evaluate(` 바로 위에 함수를 추가하고, `run()` 안의 `"model": ...` 세 줄을 한 줄로 바꿉니다.

```python
def model_label(m: dict) -> str:
    """실험 기록표의 model 칸 (예: yolo:yolov8s.pt, two_stage:resnet18)."""
    if m["family"] == "yolo":
        return f"yolo:{m['weights']}"
    if m["family"] == "rcnn":
        return f"rcnn:{m['arch']}"
    return f"{m['family']}:{m.get('classifier', '')}"


# run() 안: 전
        "model": f"yolo:{cfg['model']['weights']}"
        if cfg["model"]["family"] == "yolo"
        else f"rcnn:{cfg['model']['arch']}",
# 후
        "model": model_label(cfg["model"]),
```

**④ 새 설정 파일 `configs/two_stage.yaml`**: `detector_exp`는 내 PC의 `outputs/`에 있는 탐지 실험 이름으로 바꿉니다. data 블록(데이터 버전·class\_map)은 그 탐지 실험과 같아야 합니다.

```yaml
# 실험 M4 (모델): 탐지 + 분류 2단계. 1단계는 이미 학습이 끝난 탐지 실험을 그대로 쓴다.
# 먼저: 탐지 실험(예: YOLO baseline)이 outputs/ 에 있어야 한다. data 블록은 그 실험과 같아야 한다.
base: yolo_baseline.yaml
exp_name: 1101_modelB_twostage_v0_r18
owner: modelB
note: YOLO baseline 박스 + ResNet18 알약 분류 (M4)
model:
  family: two_stage
  detector_exp: 1028_modelA_yolov8s_v0_baseline   # ← 내 PC에 있는 탐지 실험 이름으로 바꾼다
  classifier: resnet18        # efficientnet_b0, convnext_tiny 등 torchvision 이름
  pretrained: true            # ImageNet 사전학습 (처음 실행 때 자동 다운로드)
  crop_size: 224              # 잘라낸 알약을 키울 크기
  crop_pad: 0.1               # 박스 바깥으로 10% 더 넓게 자름
  cls_weight: 0.5             # 0이면 탐지 결과 그대로, 1이면 분류 결과만
train:
  epochs: 15
  batch_size: 32
  lr: 0.0003
```

**⑤ 실행과 확인**

```bash
uv run python scripts/run_train.py --config configs/two_stage.yaml          # 분류 모델만 학습 (수십 분)
uv run python scripts/evaluate.py --exp 1101_modelB_twostage_v0_r18 --class-metrics
uv run python scripts/make_submission.py --exp 1101_modelB_twostage_v0_r18
```

학습 로그에 epoch마다 `val_acc`(잘라낸 검증 알약의 분류 정확도)가 찍힙니다. 합성 연습 데이터(사전학습 없이 3 epoch)에서 확인한 결과입니다.

```text
[two_stage] 학습 알약 140개, 검증 알약 35개, 클래스 8개
[two_stage] epoch 3: loss=0.0911 val_acc=0.857
[t_two] 검증 mAP@[0.75:0.95]=0.21823   ← 탐지 모델만 0.21449
```

**값 고르기**: `cls_weight`를 0.3 / 0.5 / 0.8로 바꿔 보는 데는 재학습이 필요 없습니다. `outputs/<실험>/config.yaml`의 값만 고치고 `evaluate.py`를 다시 돌립니다. `--tta`를 주면 1단계 탐지에 TTA가 적용됩니다. 1단계를 더 좋은 탐지 실험으로 바꾸면 분류 모델은 다시 학습하지 않고 `detector_exp`만 바꿔도 됩니다(같은 class\_map일 때).

### M5. 큰 모델 + 해상도 1024 (모델 A)

M1(크기)과 데이터 D4(해상도)가 각각 효과가 있었다면 둘을 합칩니다. 학습 시간이 baseline의 4\~6배이고 GPU 메모리를 많이 씁니다. 메모리 부족(`CUDA out of memory`)이면 `batch_size`를 2로 줄입니다. M2에서 YOLO11보다 v8이나 YOLO26이 좋았다면 `weights`를 그 버전의 m 크기로 바꿉니다.

```yaml
# configs/yolo11m_1024.yaml
# 실험 M5 (모델): 큰 모델 + 높은 해상도 (학습 시간 약 4~6배, GPU 메모리 많이 씀)
base: yolo_baseline.yaml
exp_name: 1102_modelA_yolo11m_v0_img1024
owner: modelA
note: YOLO11m + 입력 1024 (M5). 메모리가 부족하면 batch_size를 2로
model:
  weights: yolo11m.pt
data:
  img_size: 1024
train:
  batch_size: 4
```

```bash
uv run python scripts/run_train.py --config configs/yolo11m_1024.yaml --epochs 1 --exp-name test_m5
uv run python scripts/run_train.py --config configs/yolo11m_1024.yaml
```

### T3. 학습 설정 자동 탐색 (모델 A, GPU에서 반나절 이상)

사람이 하나씩 바꾸던 학습률·손실 가중치·증강 값을 Ultralytics가 자동으로 조금씩 바꿔 가며 짧게 학습해 봅니다. 반복 30번 × 15 epoch이면 baseline 학습 약 9번 분량입니다. 밤새 돌려 두기 좋은 실험입니다.

**① 새 파일 `scripts/tune_yolo.py`** (전체를 그대로 복사, X0 병합 후)

```python
"""YOLO 학습 설정 자동 탐색 (실험 T3, 모델 A). 시간이 많이 든다 (짧은 학습 x 반복 횟수).

Ultralytics의 model.tune()이 학습률·손실 가중치·증강 값을 조금씩 바꿔 가며 짧게 학습하고,
가장 좋았던 값을 best_hyperparameters.yaml로 남긴다. 그 값을 설정 파일의 yolo_args에 옮겨 본 학습한다.
(Ultralytics의 기준 점수는 0.1 x mAP50 + 0.9 x mAP50-95로 대회 지표와 다르지만 방향은 비슷하다)

실행:
    uv run python scripts/export_yolo.py --config configs/yolo_baseline.yaml   # data.yaml 준비
    uv run python scripts/tune_yolo.py --config configs/yolo_baseline.yaml --epochs 15 --iterations 30

결과: outputs/tune_<실험 이름>/tune/best_hyperparameters.yaml, tune_results.ndjson  (커밋하지 않음)
"""

from __future__ import annotations

import argparse

from src.interfaces import OUTPUT_DIR
from src.utils import apply_overrides, get_device, load_config, yolo_data_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=15, help="반복 한 번의 학습 epoch")
    ap.add_argument("--iterations", type=int, default=30, help="몇 가지 설정을 시도할지")
    ap.add_argument("--set", nargs="*", default=None, help="설정 값 바꾸기")
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = apply_overrides(load_config(args.config), args.set)
    data_yaml = yolo_data_yaml(cfg)
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} 이 없습니다. 먼저 export_yolo.py를 실행하세요.")
    dev = get_device(cfg["train"].get("device"))
    out = OUTPUT_DIR / f"tune_{cfg['exp_name']}"
    model = YOLO(cfg["model"]["weights"])
    tune_args = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "iterations": args.iterations,
        "imgsz": cfg["data"]["img_size"],
        "batch": cfg["train"]["batch_size"],
        "workers": cfg["train"].get("num_workers", 4),
        "device": dev.index if dev.type == "cuda" else dev.type,
        "optimizer": cfg["train"].get("optimizer", "auto"),
        "project": str(out),
        "name": "tune",
        "plots": False,
        "save": False,
        "val": True,
    }
    tune_args.update(cfg.get("yolo_args") or {})  # 설정 파일의 yolo_args가 이긴다 (M0과 같은 방식)
    tune_args.update({"plots": False, "save": False})  # 반복마다 그림·가중치를 남기지 않음
    model.tune(**tune_args)
    print(
        f"\n결과: {out}/tune/best_hyperparameters.yaml 의 값을 새 설정 파일 yolo_args에 옮겨 본 학습하세요."
    )


if __name__ == "__main__":
    main()
```

**② 실행**: 먼저 `--epochs 1 --iterations 2`로 오류가 없는지 본 뒤 본 탐색을 돌립니다(합성 데이터에서 이 명령으로 `best_hyperparameters.yaml`이 생기는 것을 확인).

```bash
uv run python scripts/tune_yolo.py --config configs/yolo_baseline.yaml --epochs 1 --iterations 2   # 확인용
uv run python scripts/tune_yolo.py --config configs/yolo_baseline.yaml --epochs 15 --iterations 30
```

**③ 결과 옮기기**: `outputs/tune_<실험>/tune/best_hyperparameters.yaml`을 열어 값(`lr0`, `box`, `cls`, `hsv_s` 등)을 새 설정 파일에 옮깁니다. `lr0`는 팀 설정에서는 `train.lr`이고, 나머지는 `yolo_args` 아래에 씁니다. `optimizer: auto`면 lr이 무시되므로(T1 참고) `optimizer`도 함께 적습니다.

```yaml
# configs/yolo_tuned.yaml (값은 예시. 내 탐색 결과로 바꾼다)
base: yolo_baseline.yaml
exp_name: 1103_modelA_yolov8s_v0_tuned
owner: modelA
note: tune_yolo.py 30회 탐색 결과 (T3)
train:
  optimizer: AdamW
  lr: 0.00123
yolo_args:
  box: 8.9
  cls: 0.62
  hsv_s: 0.45
```

### T4. 시드·fold 앙상블 (모델 A·B, X0 병합 후)

같은 최고 설정을 시드만 바꿔 3번 학습하면 모델마다 틀리는 곳이 조금씩 달라 WBF(P2)로 합칠 때 점수가 오르는 경우가 많습니다. 데이터 D7의 fold 모델 5개도 같은 방식으로 합칩니다.

```bash
# (가) 시드 3개: 검증셋이 같아 검증 점수로 비교 가능
uv run python scripts/run_train.py --config configs/<최고>.yaml --exp-name 1103_modelA_best_s1 --set seed=1
uv run python scripts/run_train.py --config configs/<최고>.yaml --exp-name 1103_modelA_best_s2 --set seed=2
uv run python scripts/evaluate.py --exp <최고 실험> --ensemble 1103_modelA_best_s1 1103_modelA_best_s2 --method wbf

# (나) fold 5개 (데이터 D7): 각 fold가 검증셋 일부를 학습해 검증 점수는 믿을 수 없음 → 제출 점수로 판단
uv run python scripts/make_submission.py --exp 1103_modelA_yolov8s_cv_f1 1103_modelA_yolov8s_cv_f2 \
    1103_modelA_yolov8s_cv_f3 1103_modelA_yolov8s_cv_f4 1103_modelA_yolov8s_cv_f5 --method wbf
```

합칠 모델은 같은 class\_map을 써야 합니다(데이터 D6의 v3all과 다른 버전을 섞지 않음). 모델 수가 늘수록 예측 시간이 비례해 늘어납니다.

### P5. 최종 제출: 전체 데이터 학습 + 앙상블 (Experimentation Lead, 11/4 동결 직전)

검증으로 고른 최고 설정을 **검증셋까지 포함한 전체 232장**으로 한 번 더 학습합니다(데이터 D7의 `v0_full` 분할). 검증 점수는 의미가 없으므로 조기 종료를 끄고, 최고 실험이 멈춘 epoch 수 근처로 `train.epochs`를 정합니다.

```yaml
# configs/yolo_full.yaml
# 실험 D7 (데이터): 최종 제출용. 검증셋까지 모두 학습에 쓴다 (먼저: scripts/make_folds.py)
# 검증 점수는 학습에 쓴 사진이라 의미가 없다. 설정은 검증으로 고른 최고 설정을 base로 둔다.
base: yolo_baseline.yaml       # ← 최고 설정 파일로 바꾼다 (예: yolo11s.yaml)
exp_name: 1104_modelA_yolov8s_v0full_final
owner: modelA
note: train+val 전체 학습, 최종 제출용 (D7)
data:
  version: v0_full
yolo_args:
  patience: 0                  # 조기 종료 끔 (검증이 의미 없으므로 정한 epoch까지)
```

```bash
uv run python scripts/export_yolo.py --config configs/yolo_full.yaml
uv run python scripts/run_train.py --config configs/yolo_full.yaml
# 후보 A: 전체 학습 모델 하나 (+TTA)
uv run python scripts/make_submission.py --exp 1104_modelA_yolov8s_v0full_final --tta
# 후보 B: 전체 학습 모델 + 다른 모델(R-CNN 또는 fold 모델) WBF
uv run python scripts/make_submission.py --exp 1104_modelA_yolov8s_v0full_final <다른 실험> --method wbf --weights 2 1
```

두 후보의 Kaggle 점수를 비교해 최종 제출을 고릅니다(제출은 PM(X)).

### 학습에 없는 클래스

대회 안내대로 테스트에는 학습에 없는 알약이 있습니다. 모델은 본 적 없는 클래스 번호를 만들 수 없으므로, 그 알약의 점수는 **모델 개선으로는 올릴 수 없고** 외부 데이터(데이터 가이드 D6)로만 올릴 수 있습니다. 모델 담당이 할 수 있는 것은 두 가지입니다.

- **오답을 덜 내기**: 모르는 알약을 비슷한 다른 알약으로 높은 점수로 예측하면 그 다른 클래스의 점수가 깎입니다. 04 오분류 분석에서 테스트와 가장 비슷한 사례를 보고, M4의 분류 확률이 낮을 때 점수를 낮추는 방식을 검토합니다.
- **외부 데이터가 들어오면 빠르게 재학습**: 클래스 수가 바뀌므로 지금까지 찾은 가장 좋은 설정(버전·크기·예측 방식)을 그대로 새 데이터에 적용할 수 있게 설정 파일로 정리해 둡니다.

## 최종 제출 조합 만들기

11/4 코드 동결 전까지 아래 순서로 쌓아 갑니다. 각 단계는 검증 점수가 오른 것만 다음 단계로 가져갑니다.

1. **최고 단일 YOLO 정하기**: M2·M1 → 데이터 가이드의 D1\~D5 중 효과 있던 것 → T2·M3·T1을 하나씩 얻어 설정 파일 하나로 합칩니다.
2. **두 번째 모델 준비**: R-CNN(`rcnn_balance` + `rcnn_aug`) 또는 해상도가 다른 YOLO.
3. **예측 방식 정하기**: 각 모델에 P1(NMS) → P4(TTA) → 두 모델 P2(WBF)를 검증 점수로 비교합니다.
4. **제출 파일**: PM(X)이 Kaggle에 제출합니다. Kaggle은 하루 제출 횟수를 제한하는 경우가 많으므로(대회 Overview에서 확인) 검증 점수가 가장 높은 후보 2\~3개만 올립니다.

```bash
# 예: 최고 YOLO + R-CNN, WBF, YOLO에 TTA
uv run python scripts/evaluate.py      --exp <최고YOLO> --tta --ensemble <최고RCNN> --method wbf
uv run python scripts/make_submission.py --exp <최고YOLO> <최고RCNN> --method wbf --weights 2 1 --tta
```

**검증과 Kaggle 점수가 다르게 움직일 때**: 검증셋은 수십 장이고 테스트는 842장에 학습에 없는 클래스까지 있어 순위가 뒤바뀔 수 있습니다. Kaggle Public 점수는 03 노트북 마지막 셀의 제출 기록표에 적고, 두 점수를 함께 보고 최종 후보를 고릅니다.

### 실험 기록표 (각자 채워 주세요)

| ID | 설정 / 명령 | 비교 기준 | mAP@\[0.75:0.95\] | 변화 | 판단 | 담당 |
| --- | --- | --- | --- | --- | --- | --- |
| T2 | `yolo_clspw.yaml` | YOLO baseline |  |  |  |  |
| T2 | `rcnn_balance.yaml` | R-CNN baseline |  |  |  |  |
| M1 | `yolo_v8m.yaml` | YOLO baseline |  |  |  |  |
| M2 | `yolo11s.yaml` | YOLO baseline |  |  |  |  |
| M2 | `yolo26s.yaml` | YOLO baseline |  |  |  |  |
| M3 | `yolo_box_gain.yaml` | YOLO baseline |  |  |  |  |
| T1 | `yolo_long.yaml` | YOLO baseline |  |  |  |  |
| R1 | `rcnn_select_best.yaml` | R-CNN baseline |  |  |  |  |
| P1 | `evaluate.py --nms-iou 0.7` | 같은 실험 기본 예측 |  |  |  |  |
| P4 | `evaluate.py --tta` | 같은 실험 기본 예측 |  |  |  |  |
| P2 | `evaluate.py --ensemble … --method wbf` | 두 모델 중 높은 쪽 |  |  |  |  |

**판단 기준**: 검증셋이 작아 점수가 우연히 흔들립니다. mAP@\[0.75:0.95\]가 0.01 이상 오르면 “효과 있음”, 그보다 작으면 “불분명”으로 적습니다.

### PR에 담을 것

- 내가 만든 설정 파일·고친 코드 파일(단계별로 PR 하나), 위 표의 해당 행(숫자)과 판단 한 줄
- PR 전에 `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q` 세 가지가 모두 통과해야 합니다
- 코드를 바꾸지 않는 P1과, 코드 준비가 끝난 뒤의 P4·P2 점수 비교는 PR 대신 실험 이슈에 점수를 남깁니다
- 넣지 않을 것: `outputs/`, `.pt`, 제출 CSV, 예측·학습 그림(`train_batch*.jpg`, `val_batch*_pred.jpg`) — 공개 레포입니다

점수가 오르지 않은 실험도 남깁니다. “YOLO26은 이 데이터에서 v8보다 낮았다” 같은 결과도 보고서의 비교 근거가 됩니다.
