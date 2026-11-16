# kw-vip 개발 가이드 — 구현 내용 · 실행 · 학습/평가 · 산출물 · Git 협업

2026-11-16 기준. 설계 배경과 단계별 통과 조건은 [kw-vip과제_구현개발프로세스.md](kw-vip과제_구현개발프로세스.md), 과제 규정은 [kw-vip과제.md](kw-vip과제.md) 참고.

목차
1. [구현 내용](#1-구현-내용)
2. [서버에서 실행하는 방법](#2-서버에서-실행하는-방법)
3. [과제 수행 및 코드 변경 방법](#3-과제-수행-및-코드-변경-방법)
4. [노트북에서 실행하는 방법](#4-노트북에서-실행하는-방법)
5. [학습 및 평가 프로세스](#5-학습-및-평가-프로세스)
6. [산출물 확인](#6-산출물-확인)
7. [Git 접속과 PR 하는 방법](#7-git-접속과-pr-하는-방법)
8. [자주 생기는 문제](#8-자주-생기는-문제)
9. [앞으로 남은 작업](#9-앞으로-남은-작업)

---

## 1. 구현 내용

### 1.1 전체 흐름

명령 하나가 처리되는 순서입니다. 실행부(IK 또는 ACT)는 해석 결과 JSON 만 보고 움직입니다.

```
"2층 눌러줘"
  → nlu/rules.py      부정·대기 표현 먼저 차단, 층수 추출 (숫자·한글 숫자·서수·버튼 색)
  → nlu/llm_ollama.py Ollama 로컬 LLM 이 JSON 으로 변환 (없거나 실패하면 규칙 결과 사용)
  → nlu/validate.py   장면에 없는 층 → reject, 층 없음/충돌 → clarify(되묻기)
  → {"intent": "press", "floor": 2, "reason": ...}
  → perception/       목표 버튼 앞면 위치 + 누르는 방향 (시뮬레이터 좌표 또는 카메라)
  → control/ 또는 policy/   IK 상태 기계 또는 ACT 정책이 ctrl 출력 → MuJoCo 물리 스텝
  → sim/scene.py judge()    목표 버튼 변위 ≥ 10mm, 다른 버튼 < 3mm 이면 성공
```

intent 는 4가지입니다: `press`(누르기, 움직이는 유일한 경우) / `wait`(대기·금지) / `reject`(수행 불가) / `clarify`(되묻기).

### 1.2 폴더와 파일

| 경로 | 역할 |
|---|---|
| `models/scene_v2.xml` | 버튼 3개(1층 빨강·2층 초록·3층 파랑, 8cm 간격), `press_tip` site, 큐브 5cm, `place_zone`. 원본은 `models/orig/`, 생성 스크립트 `models/make_scene_v2.py` |
| `sim/scene.py` | `make_scene(seed, condition, floor)` — 시드만으로 패널 위치·목표 층 재현. `judge()` 성공 판정 |
| `sim/recorder.py` | 관찰 카메라 + head_rgb + wrist_rgb 화면에 명령·해석·상태·버튼 변위를 겹쳐 mp4 저장 (한글 폰트 `assets/fonts/NanumGothic`) |
| `control/ik.py` | press_tip 기준 DLS IK (위치 3 + 접근축 2 구속). 다중 초기값, 이전 해에서 이어 풀기 |
| `control/press_fsm.py` | 상태 기계: CLOSE_GRIPPER → PERCEIVE → PLAN → APPROACH(5cm 앞) → PRESS(16mm, 8단계) → RETRACT. `execute(task, sim)` |
| `nlu/` | `schema.py`(JSON 형식), `rules.py`, `llm_ollama.py`, `validate.py`, `parse.py`(파이프라인), `testset_ko.yaml`(40문장 초안) |
| `perception/site_gt.py` | 시뮬레이터 좌표로 목표 위치 (기준선) |
| `perception/camera.py` | head_rgb 색 검출 + depth 복원 + 패널 평면 맞춤 → 앞면 중심·법선. 위치 오차 중앙값 1.27mm |
| `policy/gen_demos.py` | IK 상태 기계를 전문가로 데모 생성 (npz) |
| `policy/to_lerobot.py` | npz → LeRobot 데이터셋 |
| `policy/train_act.sh` | LeRobot 0.7.0 `lerobot-train` 으로 ACT 학습 |
| `policy/act_runner.py` | ACT 체크포인트 로드·실행 (20Hz, 고정 10초) |
| `pipeline.py` | 한 시도 = 해석 → 실행 → 판정 → 녹화. run_demo 와 run_eval 이 공유 |
| `run_demo.py` | 텍스트 명령 데모 (한 줄 실행 또는 대화형) |
| `eval/eval_config.yaml` | 평가 시드·조건·명령 5개 고정 |
| `eval/run_eval.py`, `eval/summarize.py` | 고정 시드 평가, 지표 집계·비교표 |
| `tests/` | 단계별 점검 (아래 5장) |
| `scripts/vla-shell-kwvip.sh` | 서버 실행 환경 (vla-shell 대체) |
| `scripts/setup_venv.sh`, `scripts/check_env.sh` | 파이썬 환경 생성·점검 |
| `docker/Dockerfile` | 서버 실행용 경량 이미지 `kw-vip-lab:x86` |

### 1.3 실행부 3가지 (같은 시드로 비교)

| `--runner` | 목표 위치 | 동작 생성 |
|---|---|---|
| `ik_gt` | 시뮬레이터 좌표 | IK 상태 기계 (기준선) |
| `ik_cam` | 카메라(RGB+depth) | IK 상태 기계 |
| `act` | (정책이 이미지로 판단) | ACT 정책 |

텍스트가 ACT 에 영향을 주는 방식: 해석된 목표 층을 원-핫 3차원으로 바꿔 관절 상태 7차원 뒤에 붙인 `observation.state`(10차원)로 넣습니다.

### 1.4 조건 정의 (eval_config.yaml)

| 조건 | 패널 이동 | 시드 |
|---|---|---|
| 학습·개발 `train` | x ±2cm, z ±2cm | 0~999 |
| 기본 평가 `base` | 같은 범위 | 10000~10009 |
| 변화 평가 `variation` | x −6~+3cm, 그리고 z ±5~8cm 또는 패널이 4~8cm 멀어짐 | 20000~20009 |
| 명령 점검 | 기본 범위 | 30000~30004 |

---

## 2. 서버에서 실행하는 방법

서버에는 **jspark 계정으로만** 접속합니다. 실행은 원래의 `vla-shell` 대신 저장소의 **`scripts/vla-shell-kwvip.sh`** 를 씁니다
(GPU + MuJoCo EGL 컨테이너, 파이썬 패키지는 저장소의 `.venv`). 결과·캐시는 모두 저장소 안(`outputs/`, `.cache/`)에 저장됩니다.

### 2.1 처음 1회

```bash
cd /home/jspark/kw-vip
scripts/setup_venv.sh act          # .venv 생성 (이미 있으면 생략)
scripts/vla-shell-kwvip.sh bash scripts/check_env.sh   # mujoco / torch / lerobot / EGL / ollama 점검
```

### 2.2 실행

저장소 루트(`/home/jspark/kw-vip`)에서:
```bash
scripts/vla-shell-kwvip.sh                                      # 대화형 셸 (나갈 때 exit)
scripts/vla-shell-kwvip.sh python run_demo.py "2층 눌러줘"        # 명령 하나 실행
scripts/vla-shell-kwvip.sh python run_demo.py "엘리베이터 3층을 선택해 줘" --seed 3 --condition variation --runner ik_cam
```
이 문서의 다른 `python ...` 명령도 앞에 `scripts/vla-shell-kwvip.sh` 를 붙이거나, 대화형 셸 안에서 실행합니다.

`run_demo.py` 옵션

| 옵션 | 뜻 |
|---|---|
| (명령 생략) | 대화형. 모호하면 "몇 층을 눌러 드릴까요?" 로 되물음 |
| `--seed N` | 장면 시드 |
| `--condition none/train/base/variation` | 패널 이동 조건 |
| `--parser rules / rules+llm` | 규칙만 / 규칙 + Ollama (Ollama 가 없으면 자동으로 규칙 사용, 로그에 `rules(fallback)`) |
| `--runner ik_gt / ik_cam / act` | 실행부 |
| `--act-ckpt 경로` | ACT 체크포인트 (`--runner act` 일 때) |
| `--no-video`, `--json` | 영상 생략, 결과 JSON 출력 |

참고: 서버에는 아직 Ollama 가 없어서 명령 해석은 규칙 파서로 동작합니다. 이미지 `kw-vip-lab:x86` 이 지워져도 스크립트가 `outputs/images/` 백업이나 `docker/Dockerfile` 로 자동 복구합니다.

---

## 3. 과제 수행 및 코드 변경 방법

### 3.1 역할과 담당 파일

| 역할 | 주로 고치는 곳 |
|---|---|
| ① 시뮬레이션·환경 | `models/`, `sim/`, `scripts/`, `docker/` |
| ② 제어 | `control/`, `policy/gen_demos.py` |
| ③ 명령 해석·인식 | `nlu/`, `perception/` |
| ④ 평가·학습·발표 | `eval/`, `policy/` (학습), `tests/` |

### 3.2 바꾸면 안 되는 약속 (팀 합의 없이 변경 금지)

- 명령 JSON: `{"intent": press|wait|reject|clarify, "floor": int|null, "reason": str}` (`nlu/schema.py`)
- 실행 인터페이스: `execute(task, sim) -> {success, peak_mm, other_mm, fail_type, t_task, ...}` (`control/press_fsm.py`)
- 성공 기준(10mm / 3mm)과 `eval/eval_config.yaml` 의 평가 시드. **평가 시드(10000 이상)는 학습·튜닝에 쓰지 않습니다.**

### 3.3 자주 하는 변경

| 하고 싶은 것 | 고칠 곳 | 확인 방법 |
|---|---|---|
| 새 표현 인식 (예: "꼭대기") | `nlu/rules.py` 패턴 또는 `nlu/llm_ollama.py` 의 few-shot | `python tests/eval_nlu.py -v` |
| 테스트 문장 추가 | `nlu/testset_ko.yaml` (팀원 각 10문장으로 교체할 것) | 같음 |
| 누르기 깊이·속도 | `control/press_fsm.py` 의 `PressConfig` | `python tests/phase_checks.py phase1` |
| 패널 변화 범위 | `sim/scene.py` 의 `sample_panel_offset` | `python tests/phase_checks.py reach` |
| 카메라 인식 개선 | `perception/camera.py` | `python tests/check_camera.py` |
| 장면 XML 변경 | `models/make_scene_v2.py` 수정 후 재생성 (아래) | `python tests/phase_checks.py phase0` |
| 영상 화면 구성 | `sim/recorder.py` | `run_demo.py` 로 영상 확인 |

장면 재생성:
```bash
python models/make_scene_v2.py models/orig/wheelchair_piper_lite.xml models/scene_v2.xml
```
XML 을 바꾸면 발표에서 설명해야 하므로 변경 내용을 PR 설명에 적어 주세요.

### 3.4 변경 후 최소 점검 (PR 전에)

```bash
scripts/vla-shell-kwvip.sh python tests/phase_checks.py       # PHASE0 PASS, PHASE1 PASS 확인
scripts/vla-shell-kwvip.sh python tests/eval_nlu.py -v         # 금지/불가 오동작 0 확인
scripts/vla-shell-kwvip.sh python tests/test_nlu_llm_mock.py   # ALL PASS (Ollama 없이 LLM 경로 점검)
```

---

## 4. 노트북에서 실행하는 방법

과제 규정상 **최종 추론과 시연은 각자 노트북에서, 서버 연결 없이** 해야 합니다.

### 4.1 설치

Python 3.10 이상 (서버는 3.12) 필요.
```bash
git clone <원격 저장소 주소> kw-vip && cd kw-vip
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt          # IK·카메라·규칙 파서 (GPU 불필요)
pip install -r requirements-act.txt      # ACT 까지 쓸 때 (torch 포함, 용량 큼)
```

### 4.2 Ollama (명령 해석 LLM)

1. https://ollama.com 에서 설치
2. 모델 받기: `ollama pull qwen2.5:3b` (후보: `qwen2.5:1.5b`, `llama3.2:3b`. `tests/eval_nlu.py` 로 비교해 팀에서 하나로 정함)
3. 다른 모델을 쓰면 환경 변수 `KWVIP_LLM_MODEL=모델이름`

### 4.3 ACT 체크포인트 복사

서버의 `/home/jspark/kw-vip/outputs/train/act_press_v1_s20000/checkpoints/020000/pretrained_model` 폴더(약 198MB)를 노트북의 `checkpoints/act_press_v1/` 로 복사합니다.
```bash
scp -r jspark@<서버주소>:/home/jspark/kw-vip/outputs/train/act_press_v1_s20000/checkpoints/020000/pretrained_model checkpoints/act_press_v1
```

### 4.4 실행 (네트워크를 끊고)

```bash
# Linux 는 화면이 있으면 MUJOCO_GL 지정 없이 실행. 화면 없는 Linux 만 MUJOCO_GL=egl
# Windows/macOS 는 MUJOCO_GL 을 지정하지 않음
python run_demo.py "2층 눌러줘"                                   # 결과 영상: outputs/videos/
python run_demo.py                                               # 대화형
python run_demo.py "3층 버튼 눌러줘" --runner act --act-ckpt checkpoints/act_press_v1
```
- GPU 가 없으면 ACT 는 자동으로 CPU 에서 돕니다.
- Windows 에서 `bash` 스크립트(`scripts/*.sh`, `policy/train_act.sh`)는 쓰지 않습니다. 노트북에서는 데모·평가·지연 측정만 하면 됩니다.

### 4.5 노트북에서 측정해 발표에 넣을 것

- 추론 지연: `python eval/run_eval.py --runner act --parser rules+llm --run act_laptop --act-ckpt checkpoints/act_press_v1` → `summary.json` 의 `llm_ms_median`, `policy_ms_median`, trials.jsonl 의 `policy_ms_max`
- 노트북 사양: OS, CPU, GPU, RAM
- 발표 영상: 성공 1개(변화 조건이면 더 좋음), 실패 1개(원인이 보이는 것), 명령 거절 1개

---

## 5. 학습 및 평가 프로세스

모든 명령은 jspark 로 저장소 루트에서, `scripts/vla-shell-kwvip.sh` 를 앞에 붙이거나 대화형 셸 안에서 실행합니다. 결과는 `outputs/` 에 저장됩니다.

### 5.1 단계별 통과 조건

| 단계 | 명령 | 통과 조건 | 현재 결과 |
|---|---|---|---|
| Phase 0 장면 | `python tests/phase_checks.py phase0` | 3개 층 모두 10mm 이상 + mp4 | PASS (12.0~12.9mm) |
| Phase 1 IK | `python tests/phase_checks.py phase1` | 학습 범위 30회 중 28회 이상 | PASS (30/30) |
| 변화 범위 도달성 | `python tests/phase_checks.py reach` | (사전 확인용) | 49/50 (1회 IK 도달 불가) |
| Phase 2 명령 해석 | `python tests/eval_nlu.py -v` | 규칙만 / 규칙+LLM 정답 수 표, 금지·불가 오동작 0 | 규칙 36/40, 오동작 0 (LLM 미측정) |
| Phase 3 기준선 평가 | `python eval/run_eval.py --runner ik_gt` | IK 결과표 첫 버전 | 20/20, 명령 5/5 |
| Phase 4 ACT | 아래 5.2 | ACT 20회 결과와 IK 비교표 | 기본 9/10, 변화 1/10 |
| Phase 5 카메라 | `python eval/run_eval.py --runner ik_cam` | 같은 20회 + 위치 오차 | 20/20, 오차 1.27mm |

### 5.2 ACT 학습 순서

```bash
# 1) 데모 생성 (IK 전문가, 개발 시드만 사용, 실패 데모는 자동 제외)
python policy/gen_demos.py --per-floor 10 --out outputs/datasets/press_v1_small              # 형식 확인용
python policy/gen_demos.py --per-floor 80 --seed-start 100 --out outputs/datasets/press_v1   # 본 데이터 (약 2분 20초)

# 2) LeRobot 형식으로 변환
python policy/to_lerobot.py --src outputs/datasets/press_v1 --dst outputs/lerobot/press_v1

# 3) 학습 (인자: 데이터셋 이름, 스텝 수, 배치 크기)
bash policy/train_act.sh press_v1_small 3000 8      # 과적합 확인 (약 2분)
bash policy/train_act.sh press_v1 20000 32          # 본 학습 (약 21분, 4070 Ti)

# 4) 체크포인트 선택 — 개발 시드로만 (평가 시드 사용 금지)
python tests/act_check.py outputs/train/act_press_v1_s20000/checkpoints/013332/pretrained_model --seeds 500 530
python tests/act_check.py outputs/train/act_press_v1_s20000/checkpoints/020000/pretrained_model --seeds 500 530

# 5) 선택한 체크포인트로 평가
python eval/run_eval.py --runner act --parser rules --run act_rules \
    --act-ckpt outputs/train/act_press_v1_s20000/checkpoints/020000/pretrained_model
```

발표에 공개할 학습 정보 (현재 값)

| 항목 | 값 |
|---|---|
| 에피소드 수 | 240 (층당 80), 42,212 프레임 |
| 데이터 생성 | IK 상태 기계 전문가, 30% 에피소드에 관절 교란(0.15rad) 후 복구, 실패 0건 |
| 관측 | head_rgb, wrist_rgb (120×160) + 관절 7 + 목표 층 원-핫 3 |
| 행동 | ctrl 7 (관절 목표각 6 + 그리퍼), 20Hz (물리 25스텝 유지) |
| 학습 | LeRobot 0.7.0 ACT, 배치 32, 20,000 스텝, RTX 4070 Ti, 21.5분 |
| 체크포인트 | 198MB |
| 사전학습 가중치 | ResNet18 백본 (torchvision ImageNet, ACT 기본 설정) |
| 체크포인트 선택 | 개발 시드 500~529: 013332 → 29/30, 020000 → 29/30 (동률이라 마지막 선택) |

### 5.3 평가

```bash
python eval/run_eval.py --runner ik_gt  --parser rules+llm      # 기준선
python eval/run_eval.py --runner ik_cam --parser rules+llm      # 카메라 인식
python eval/run_eval.py --runner act    --parser rules+llm --act-ckpt <체크포인트>
python eval/summarize.py outputs/eval/*/trials.jsonl    # 발표용 비교표 (마크다운)
```
- 한 번 실행에 기본 10회 + 변화 10회 + 명령 5회, 모든 시도의 영상이 저장됩니다. `--no-video` 로 생략 가능.
- 시드와 조건은 `eval/eval_config.yaml` 에 평가 전에 고정하고, 실패한 시도도 빼지 않습니다.
- 실패 유형: `nlu`(해석) / `perception`(인식) / `ik`(도달 불가) / `timeout`(시간 안에 목표 미달, ACT 는 버튼 미접촉 포함) / `contact`(10mm 미만) / `wrong_button`(다른 버튼 3mm 이상) / `moved_when_forbidden`

### 5.4 현재 비교표 (서버, 파서 = rules)

| 실행부 | 기본 SR | 변화 SR | 전체 SR | 일반화 격차 | 성공 시 시간(중앙값) | 명령 적합성 | 주요 실패 유형 |
|---|---|---|---|---|---|---|---|
| IK + 시뮬레이터 좌표 (기준선) | 100% | 100% | 100% | 0%p | 7.5s | 5/5 | - |
| IK + 카메라 인식 | 100% | 100% | 100% | 0%p | 7.5s | 5/5 | - |
| ACT | 90% | 10% | 50% | 80%p | 10.0s (고정 실행 시간) | 5/5 | timeout 8, contact 2 |

해석: ACT 는 학습 범위(±2cm) 안에서는 잘 누르지만, 패널이 5cm 이상 옮겨지면 버튼에 닿지 못합니다. 개선 방향은 학습 범위 확대, 또는 카메라로 찾은 목표 위치를 정책 입력에 넣는 하이브리드입니다. 남은 일: Ollama 설치 후 `rules+llm` 으로 다시 측정, 노트북에서 지연 측정.

---

## 6. 산출물 확인

### 6.1 위치 (모두 `/home/jspark/kw-vip/outputs/`, git 에는 올라가지 않음)

| 산출물 | 경로 |
|---|---|
| 데모 영상 | `outputs/videos/` |
| 평가 시도별 로그 | `outputs/eval/<run>/trials.jsonl` |
| 평가 요약 | `outputs/eval/<run>/summary.json` |
| 평가 영상 | `outputs/eval/<run>/videos/<run>_<seed>.mp4` |
| 명령 해석 결과 | `outputs/nlu/nlu_report.json` |
| 데모 데이터 (npz) | `outputs/datasets/press_v1*` |
| LeRobot 데이터셋 | `outputs/lerobot/press_v1*` |
| ACT 체크포인트 (평가에 쓴 것) | `outputs/train/act_press_v1_s20000/checkpoints/020000/pretrained_model` |
| 실행 이미지 백업 | `outputs/images/kw-vip-lab_x86.tar` |

현재 `<run>`: `ik_gt_rules`, `ik_cam_rules`, `act_rules`.

### 6.2 확인 방법

```bash
# 비교표
scripts/vla-shell-kwvip.sh bash -c "python eval/summarize.py outputs/eval/*/trials.jsonl"
# 실패한 시도만 보기
grep '"success": false' outputs/eval/act_rules/trials.jsonl
# 영상을 내 컴퓨터로 가져오기 (서버에는 화면이 없음)
scp jspark@<서버주소>:/home/jspark/kw-vip/outputs/eval/act_rules/videos/act_rules_20000.mp4 .
```
VS Code Remote-SSH 를 쓰면 탐색기에서 mp4 를 오른쪽 클릭 → Download 로 받을 수 있습니다.

시도별 로그 한 줄의 주요 필드: `split`, `seed`, `command`, `parsed`, `parser`, `llm_ms`, `success`, `peak_mm`, `other_mm`, `fail_type`, `t_task_s`, `policy_ms`, `perception.err_mm`, `video`.

영상 화면: 왼쪽 큰 화면은 관찰 카메라, 오른쪽은 head_rgb / wrist_rgb. 아래 띠에 명령, 해석 JSON, 실행부와 상태, 버튼별 변위 막대(흰 선 = 10mm)가 표시되고, 끝에 SUCCESS / FAIL: 유형 / 대기(정상) / 거절(정상) 이 뜹니다.

### 6.3 발표 체크리스트와 연결

| 발표 항목 | 가져올 곳 |
|---|---|
| 20회 결과, 명령 5회, 실패 유형 | `summarize.py` 비교표, `summary.json` |
| 성공·실패·거절 영상 | `eval/<run>/videos/` (예: 실패는 `act_rules_20000.mp4`, 거절은 `*_cmd_30004.mp4`) |
| XML 변경 내용 | 구현개발프로세스 문서의 'scene_v2.xml 변경 내용' 표 |
| 학습 정보 | 5.2 표 |
| 재현 정보 | `eval/eval_config.yaml` (성공 기준, 시드, 조건) |

---

## 7. Git 접속과 PR 하는 방법

> 현재 `/home/jspark/kw-vip` 는 아직 git 저장소가 아니고 원격 저장소도 없습니다. 아래 7.1 을 jspark 가 1회 진행합니다.
> 서버에서는 jspark 계정으로 git 작업과 실행을 모두 합니다. 팀원은 각자 노트북에서 clone 해서 작업합니다.

### 7.1 저장소 만들기 (jspark, 1회)

1. GitHub 에서 빈 저장소 생성 (예: `kw-vip`, Private 권장). README 등은 추가하지 않음.
2. 팀원을 Collaborator 로 초대: 저장소 → Settings → Collaborators → Add people.
3. 서버에서:
```bash
cd /home/jspark/kw-vip
git init -b main
git add .
git commit -m "kw-vip 초기 구현: 장면, IK, 명령 해석, 카메라 인식, ACT, 평가"
git remote add origin git@github.com:<계정>/kw-vip.git
git push -u origin main
```
`.gitignore` 로 `outputs/`, `.venv/`, `*.mp4` 는 올라가지 않습니다. 체크포인트·데이터셋은 git 이 아니라 scp 로 옮깁니다.

4. (권장) main 보호: Settings → Branches → Add rule → `main` → "Require a pull request before merging" 체크. 이러면 main 에 직접 push 가 막히고 PR 로만 합쳐집니다.

### 7.2 GitHub 접속 설정 (각자 1회)

SSH 키 방식:
```bash
ssh-keygen -t ed25519 -C "<내 GitHub 이메일>"
cat ~/.ssh/id_ed25519.pub          # 출력 내용을 GitHub → Settings → SSH and GPG keys → New SSH key 에 붙여넣기
ssh -T git@github.com              # "Hi <계정>!" 이 나오면 성공
git config --global user.name  "<이름>"
git config --global user.email "<GitHub 이메일>"
```
HTTPS 를 쓰려면 비밀번호 대신 Personal Access Token 을 씁니다 (GitHub → Settings → Developer settings → Tokens).

### 7.3 작업 → PR 흐름

```bash
git clone git@github.com:<계정>/kw-vip.git && cd kw-vip     # 처음 1회
git switch main && git pull                                # 작업 시작 전 최신화
git switch -c feat/nlu-testset                             # 브랜치 생성 (이름: feat/..., fix/..., exp/...)
# ... 코드 수정 ...
python tests/phase_checks.py && python tests/eval_nlu.py   # 3.4 최소 점검
git add -p                                                 # 바꾼 부분 확인하며 추가
git commit -m "nlu: 팀원 테스트 문장 40개 반영"
git push -u origin feat/nlu-testset
```
그다음 GitHub 저장소 페이지에 뜨는 "Compare & pull request" 버튼 → 제목·설명 작성 → Reviewers 에 팀원 지정 → Create pull request.

PR 설명에 넣을 것:
- 무엇을 왜 바꿨는지
- 확인 결과 (예: `PHASE1 30/30`, `eval_nlu 38/40, 오동작 0`)
- 평가 수치가 바뀌었으면 전후 비교표
- 장면 XML, 명령 JSON, `execute()` 인터페이스, `eval_config.yaml` 을 바꿨다면 반드시 명시 (팀 합의 필요)

리뷰 후 Merge 하면 다른 사람은 `git switch main && git pull` 로 받습니다.

### 7.4 서버에 반영하기

서버의 `/home/jspark/kw-vip` 가 실행에 쓰이는 코드입니다. PR 이 합쳐지면 jspark 로:
```bash
cd /home/jspark/kw-vip && git switch main && git pull
```

### 7.5 충돌이 났을 때

```bash
git switch feat/내브랜치
git fetch origin && git merge origin/main      # 충돌 파일이 표시됨
# 파일에서 <<<<<<< ======= >>>>>>> 구간을 정리
git add <파일> && git commit && git push
```

---

## 8. 자주 생기는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| 원래 `vla-shell` 이 `pull access denied for vla-lab` 로 실패 | 그 이미지가 서버에서 지워짐 | 쓰지 않음. `scripts/vla-shell-kwvip.sh` 사용 |
| `[kw-vip] .venv 가 없습니다` | 저장소에 venv 없음 | `scripts/setup_venv.sh act` |
| 해석이 항상 `rules(fallback)` | Ollama 미설치 또는 접속 불가 | `bash scripts/check_env.sh` 로 확인, Ollama 설치 후 `ollama pull <모델>` |
| 영상 글자가 □ 로 나옴 | 한글 폰트 없음 | `assets/fonts/NanumGothic-Regular.ttf` 가 있는지 확인, 또는 `KWVIP_FONT=<폰트경로>` |
| 종료 시 `EGLError` 경고가 잔뜩 출력 | MuJoCo 렌더러 정리 시 경고 | 결과에 영향 없음, 무시 |
| `lerobot-train: unrecognized arguments` | LeRobot 버전마다 옵션 이름이 다름 | `lerobot-train --help` 로 확인 (0.7.0 은 `--wandb` 대신 `--trackio.enable`) |
| ACT 가 변화 조건에서 실패 | 학습 범위 밖 일반화 격차 | 예상된 결과. 발표의 분석 재료로 사용 |

---

## 9. 앞으로 남은 작업

2026-11-16 기준. 담당 번호는 3.1 의 역할(① 시뮬레이션·환경, ② 제어, ③ 명령 해석·인식, ④ 평가·학습·발표)입니다.

### 9.1 현재 상태 요약

| 항목 | 상태 |
|---|---|
| Phase 0~3 (장면, IK, 명령 해석, 기준선 평가) | 완료. IK 20/20, 명령 5/5 |
| Phase 4 ACT | 1차 완료. 기본 9/10, 변화 1/10 (일반화 격차 80%p) |
| Phase 5 카메라 인식 | 1차 완료. 20/20, 위치 오차 1.27mm |
| 명령 해석 LLM | **미측정** (서버에 Ollama 없음, 모두 규칙 폴백) |
| 일반화 정의·평가 | **부분만 됨**: 패널 평행이동(x, z, 멀어짐)만 있음. 회전·버튼 배치 변화 없음 |
| 노트북 이식·지연 측정·발표 영상 | 미착수 |

### 9.2 일반화 정의 (필수)

지금 '변화 조건'은 패널을 평행이동하는 것뿐입니다. 발표에서 "무엇에 대해 일반화되는가"를 설명하려면 변화 요인을 아래처럼 나눠 정의하고, 요인마다 따로 성능을 재야 합니다.

**좌표 기준**: 휠체어는 world 에 고정(+x 앞, −y 사용자 오른쪽)이므로 "휠체어 위치 변화"는 "휠체어 기준 패널의 상대 위치·방향 변화"로 구현합니다. 휠체어를 직접 옮기면 팔·카메라가 같이 움직이므로 결과는 같습니다.

| 요인 | 변수 | 현재 | 정의할 범위 (예시, Phase 0 처럼 도달성 먼저 확인 후 확정) |
|---|---|---|---|
| A. 패널 좌우·높이 | 패널 x, z | 학습 ±2cm, 변화 x −6~+3cm / z ±5~8cm | 그대로 유지 |
| B. 휠체어–패널 거리 | 패널 법선 방향 거리 (y) | 변화에서 4~8cm 멀어짐만 | 가까워짐 −3~0cm, 멀어짐 0~8cm 로 양쪽 확장. 도달 한계(가까워짐 6cm 에서 IK 실패)를 경계로 표시 |
| C. 휠체어–패널 각도 | 패널 yaw (법선과 휠체어 접근 방향 사이 각) | 없음 (항상 정면) | ±10°, ±20° 단계. 누르는 방향(`into`)이 패널 법선을 따라가는지 확인 |
| D. 버튼 순서 | 층 → 위치 매핑 | 고정: head_rgb 시점에서 왼쪽부터 3·2·1층 (패널 좌표 x = +8, 0, −8cm) | 오름차순(왼→오 1·2·3) / 내림차순(3·2·1) |
| E. 버튼 방향 | 가로 1줄 / 세로 1줄 | 가로 1줄만 | 세로 1줄(아래→위 1·2·3 또는 반대). 패널 높이 24cm 안에 8cm 간격 3개가 들어감 |
| F. 버튼 간격 | 간격 | 8cm 고정 | 6cm / 8cm / 10cm (오동작 = 인접 버튼 눌림 위험 증가) |

**주의 — 색이 층을 알려 주는 문제**: 지금은 색이 층에 묶여 있어서(1층 빨강, 2층 초록, 3층 파랑) 순서·방향을 바꿔도 카메라 인식은 색만 보고 맞힙니다. 진짜 "배치 일반화"를 보려면 다음 중 하나가 필요합니다.
- 버튼 표면에 숫자 텍스처(1·2·3)를 넣고, 색은 무작위 또는 모두 같은 색으로 → 숫자 인식 필요 (발표에서 XML 변경으로 설명)
- 또는 "색 = 층" 을 유지하되, 이것이 가정임을 발표에서 명시

**학습 분포 vs 평가 분포**: 요인마다 "학습에 포함한 값"과 "학습에 없던 값"을 미리 나눠 표로 고정합니다. 예: ACT 는 가로·오름차순만 학습 → 평가에서 내림차순·세로를 처음 봄.

구현할 곳

| 작업 | 파일 | 담당 |
|---|---|---|
| 조건에 요인 B~F 추가: `sample_panel_offset` 를 `sample_layout(rng, condition)` 로 확장 (패널 위치·yaw, 버튼 순서·방향·간격) | `sim/scene.py` | ① |
| 버튼 위치를 런타임에 바꾸기: `model.body_pos[elevator_button_i]` 를 배치에 맞게 설정, 패널 yaw 는 `model.body_quat[elevator_panel]` | `sim/scene.py` | ① |
| (선택) 숫자 텍스처·단색 버튼 | `models/make_scene_v2.py` → `scene_v3.xml` | ① |
| 카메라 인식이 배치 사전지식(가로·8cm)에 기대는 부분 일반화: 간격·방향을 가정하지 않는 조합 선택 | `perception/camera.py` 의 `detect_buttons` | ③ |
| 로그에 요인 값 기록 (`layout`: order, orientation, spacing, yaw) | `pipeline.py`, `eval/summarize.py` (요인별 집계) | ④ |

### 9.3 일반화 성능 확인 (필수)

1. **도달성 사전 확인** (평가 전, 개발 시드): 요인마다 IK + 시뮬레이터 좌표로 50회씩 돌려 도달 가능한 범위만 남깁니다. `tests/phase_checks.py reach` 를 요인별로 확장. 기준: 범위 안에서 IK 성공 ≥ 95%.
2. **평가 설정 고정**: `eval/eval_config.yaml` 에 요인별 split 추가. 시드는 겹치지 않게 새 구간을 씁니다.

| split | 내용 | 시드 (예) | 횟수 |
|---|---|---|---|
| `base` | 학습 분포 (기존) | 10000~10009 | 10 |
| `variation` | 패널 평행이동 (기존) | 20000~20009 | 10 |
| `gen_distance` | 요인 B | 50000~50009 | 10 |
| `gen_yaw` | 요인 C | 51000~51009 | 10 |
| `gen_order` | 요인 D (내림차순 등) | 52000~52009 | 10 |
| `gen_vertical` | 요인 E | 53000~53009 | 10 |
| `gen_spacing` | 요인 F | 54000~54009 | 10 |

   (기존 `variation` 10회는 과제의 필수 '변화 조건 평가'로 그대로 두고, `gen_*` 는 추가 분석입니다. 시드 40000~40049 는 도달성 확인에 이미 썼으므로 평가에 쓰지 않습니다.)
3. **실행**: 실행부 3개 × 같은 시드.
   ```bash
   python eval/run_eval.py --runner ik_gt  --splits base variation gen_distance gen_yaw gen_order gen_vertical gen_spacing
   python eval/run_eval.py --runner ik_cam --splits ...
   python eval/run_eval.py --runner act    --act-ckpt <체크포인트> --splits ...
   ```
4. **지표**: 요인별 성공률, 학습 분포 대비 격차(%p), 실패 유형(특히 `wrong_button` = 순서·간격 변화에서 인접 버튼 오동작), 카메라 위치 오차(mm). 결과는 "요인 × 실행부" 표 하나로 정리합니다.
5. **해석 포인트 (발표 5장 '한계와 개선')**
   - IK + 좌표는 정답 위치를 받으므로 도달 가능하면 거의 100% → 일반화 문제는 '인식'과 '학습 정책'에서 생긴다는 것을 보여 주는 기준선
   - ACT 는 원-핫 층 정보만 받으므로, 버튼 순서·방향이 바뀌면 "층 → 위치" 대응을 이미지에서 다시 배워야 함 → 크게 떨어질 것으로 예상
   - 개선 실험 1: 학습 데이터에 요인 값을 섞어(ACT 재학습) 격차가 줄어드는지
   - 개선 실험 2 (하이브리드): 카메라가 찾은 버튼 3D 위치를 ACT 의 상태 입력에 넣거나, 접근은 ACT·마지막 누르기는 IK

### 9.4 그 밖의 남은 작업

| # | 작업 | 내용 · 통과 조건 | 담당 |
|---|---|---|---|
| 1 | Ollama 설치와 LLM 측정 | 서버(또는 노트북)에 Ollama 설치 → 후보 모델(qwen2.5:1.5b / 3b, llama3.2:3b)을 `tests/eval_nlu.py` 로 비교해 하나 선택 → `--parser rules+llm` 으로 평가 다시 실행. 표: 규칙만 / 규칙+LLM 정답 수, 금지·불가 오동작 0, LLM 지연 중앙값 | ③ |
| 2 | 명령 테스트셋 교체 | 팀원 4명 × 10문장 = 40문장으로 `nlu/testset_ko.yaml` 교체 (재표현·금지·불가능·모호 골고루). 평가용 5문장은 넣지 않음 | 전원 → ③ |
| 3 | ACT 개선 | 학습 범위 확대(패널 ±6cm 등) 데이터로 재학습해 변화 조건 성공률 비교. 실패 유형 `timeout`(미접촉)·`contact`(깊이 부족) 감소 확인. 체크포인트 선택은 개발 시드만 | ② ④ |
| 4 | 실패 분석 | 실패 영상마다 원인 분류(인식 / 손목 방향 / 도달 / 접촉 깊이 / 시간) 표 작성, 대표 영상 선정 | 전원 |
| 5 | 노트북 이식 | 각자 clone → `requirements*.txt` 설치 → 체크포인트·Ollama 모델 복사 → 네트워크 끊고 `run_demo.py` 성공 확인 | 전원 |
| 6 | 노트북 지연 측정 | LLM 1회, ACT 1스텝(`policy_ms_max`) 지연, 노트북 CPU·GPU·RAM 기록 | ④ |
| 7 | 발표 영상 | 노트북에서 오프라인 녹화: 성공 1(변화 조건 권장), 실패 1(원인이 보이는 것), 명령 거절 1 | ④ |
| 8 | 최종 비교표 | 실행부(IK+좌표 / IK+카메라 / ACT) × (기본·변화·전체 SR, 격차, 시간, 지연, 명령 적합성, 실패 유형) + 9.3 일반화 표 | ④ |
| 9 | 발표 자료 | 5분 PPT. 과제 문서 7장 순서, 2장 체크리스트(XML 변경, 데이터·학습 정보, 노트북 추론 사실, 한계: 표본 20회·팔 충돌 비활성) | 전원 |
| 10 | Git 원격 연결 | GitHub 저장소 공개 범위 결정(강의 PDF 포함 여부 확인) → push → 팀원 초대, main 보호 설정 (7장) | jspark |
| 11 | (선택) 큐브 시나리오 B | 집기 상태 기계, 스키마에 `pick_place` 추가, 같은 방식으로 20회 평가 (구현개발프로세스 Phase 6) | ② ① |
| 12 | (선택) 팔 충돌 | 팔 링크 충돌을 켜고 충돌 횟수 측정. 안 하면 발표에서 "충돌 안전성 주장 없음"을 명시 | ① |
