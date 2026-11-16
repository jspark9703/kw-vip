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

### 2.1 현재 서버 상황 (꼭 읽기)

- 원래 `vla-shell` 은 **지금 실행되지 않습니다.** 쓰는 이미지 `vla-lab:x86` 이 서버에서 사라졌습니다. 서버의 Coolify 가 매일 00:00 에 Docker 정리 작업을 돌리는데, 이때 지워진 것으로 추정됩니다.
- `vla-shell` 은 `/srv/vla/workspace` 만 컨테이너에 연결하므로, `/home/jspark/kw-vip` 의 코드가 보이지 않습니다.
- 그래서 **`scripts/vla-shell-kwvip.sh`** 를 씁니다. vla-shell 과 같은 옵션(GPU, `/workspace`, `/cache`, `/results`, EGL)에 저장소 연결과 결과 폴더 설정을 더한 것입니다.
  - 이미지는 `vla-lab:x86` 이 있으면 그것을, 없으면 `kw-vip-lab:x86` 을 씁니다.
  - `kw-vip-lab:x86` 이 지워져도 백업(`/srv/vla/results/kw-vip/images/kw-vip-lab_x86.tar`)에서 자동으로 다시 불러옵니다.
  - 결과는 `/srv/vla/results/kw-vip` (컨테이너 안에서는 `/results/kw-vip`) 에 저장됩니다.
- 서버에 **Ollama 가 없습니다.** 명령 해석은 자동으로 규칙 파서로 동작합니다(로그에 `rules(fallback)`).

### 2.2 처음 1회 설정

jspark 계정:
```bash
cd /home/jspark/kw-vip
scripts/setup_venv.sh act                        # .venv 생성 (이미 되어 있으면 생략)
cp scripts/vla-shell-kwvip.sh /tmp/ && chmod 755 /tmp/vla-shell-kwvip.sh
```

vipuser 계정 (비밀번호는 팀 내부 공유):
```bash
mkdir -p ~/bin && cp /tmp/vla-shell-kwvip.sh ~/bin/
echo 'export KWVIP_REPO=/home/jspark/kw-vip' >> ~/.bashrc
source ~/.bashrc
```

> vipuser 는 `/home/jspark` 안을 직접 읽을 수 없어서 스크립트를 `/tmp` 를 거쳐 복사합니다. 컨테이너 안에서는 저장소가 정상으로 보입니다.
> 스크립트를 수정했다면 위 복사를 다시 해야 vipuser 쪽에 반영됩니다.

### 2.3 실행

```bash
# vipuser 로
vla-shell-kwvip.sh                                   # 대화형 셸 (저장소 루트에서 시작)
vla-shell-kwvip.sh bash scripts/check_env.sh         # 환경 점검: mujoco / torch / lerobot / EGL / ollama
vla-shell-kwvip.sh python run_demo.py "2층 눌러줘"
vla-shell-kwvip.sh python run_demo.py "엘리베이터 3층을 선택해 줘" --seed 3 --condition variation --runner ik_cam
```

`run_demo.py` 옵션

| 옵션 | 뜻 |
|---|---|
| (명령 생략) | 대화형. 모호하면 "몇 층을 눌러 드릴까요?" 로 되물음 |
| `--seed N` | 장면 시드 |
| `--condition none/train/base/variation` | 패널 이동 조건 |
| `--parser rules / rules+llm` | 규칙만 / 규칙 + Ollama |
| `--runner ik_gt / ik_cam / act` | 실행부 |
| `--act-ckpt 경로` | ACT 체크포인트 (`--runner act` 일 때) |
| `--no-video`, `--json` | 영상 생략, 결과 JSON 출력 |

### 2.4 jspark 계정에서 빠르게 확인할 때

컨테이너 없이 호스트에서도 돌아갑니다 (결과는 저장소의 `outputs/`).
```bash
cd /home/jspark/kw-vip
MUJOCO_GL=egl .venv/bin/python run_demo.py "2층 눌러줘" --parser rules
```

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
vla-shell-kwvip.sh python tests/phase_checks.py          # PHASE0 PASS, PHASE1 PASS 확인
vla-shell-kwvip.sh python tests/eval_nlu.py -v            # 금지/불가 오동작 0 확인
vla-shell-kwvip.sh python tests/test_nlu_llm_mock.py      # ALL PASS (Ollama 없이 LLM 경로 점검)
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

서버의 `/srv/vla/results/kw-vip/train/act_press_v1_s20000/checkpoints/020000/pretrained_model` 폴더(약 198MB)를 노트북의 `checkpoints/act_press_v1/` 로 복사합니다.
```bash
scp -r <서버계정>@<서버주소>:/srv/vla/results/kw-vip/train/act_press_v1_s20000/checkpoints/020000/pretrained_model checkpoints/act_press_v1
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

모든 명령은 vipuser 로 `vla-shell-kwvip.sh` 안에서 실행합니다(앞에 `vla-shell-kwvip.sh` 를 붙이거나 대화형 셸 안에서).

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
python policy/gen_demos.py --per-floor 10 --out /results/kw-vip/datasets/press_v1_small              # 형식 확인용
python policy/gen_demos.py --per-floor 80 --seed-start 100 --out /results/kw-vip/datasets/press_v1   # 본 데이터 (약 2분 20초)

# 2) LeRobot 형식으로 변환
python policy/to_lerobot.py --src /results/kw-vip/datasets/press_v1 --dst /results/kw-vip/lerobot/press_v1

# 3) 학습 (인자: 데이터셋 이름, 스텝 수, 배치 크기)
bash policy/train_act.sh press_v1_small 3000 8      # 과적합 확인 (약 2분)
bash policy/train_act.sh press_v1 20000 32          # 본 학습 (약 21분, 4070 Ti)

# 4) 체크포인트 선택 — 개발 시드로만 (평가 시드 사용 금지)
python tests/act_check.py /results/kw-vip/train/act_press_v1_s20000/checkpoints/013332/pretrained_model --seeds 500 530
python tests/act_check.py /results/kw-vip/train/act_press_v1_s20000/checkpoints/020000/pretrained_model --seeds 500 530

# 5) 선택한 체크포인트로 평가
python eval/run_eval.py --runner act --parser rules --run act_rules \
    --act-ckpt /results/kw-vip/train/act_press_v1_s20000/checkpoints/020000/pretrained_model
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
python eval/summarize.py /results/kw-vip/eval/*/trials.jsonl    # 발표용 비교표 (마크다운)
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

### 6.1 위치

| 산출물 | 서버 경로 (컨테이너 안에서는 `/results/kw-vip/...`) |
|---|---|
| 데모 영상 | `/srv/vla/results/kw-vip/videos/` |
| 평가 시도별 로그 | `/srv/vla/results/kw-vip/eval/<run>/trials.jsonl` |
| 평가 요약 | `/srv/vla/results/kw-vip/eval/<run>/summary.json` |
| 평가 영상 | `/srv/vla/results/kw-vip/eval/<run>/videos/<run>_<seed>.mp4` |
| 명령 해석 결과 | `/srv/vla/results/kw-vip/nlu/nlu_report.json` |
| 데모 데이터 (npz) | `/srv/vla/results/kw-vip/datasets/press_v1*` |
| LeRobot 데이터셋 | `/srv/vla/results/kw-vip/lerobot/press_v1*` |
| ACT 체크포인트 | `/srv/vla/results/kw-vip/train/act_press_v1_s20000/checkpoints/` |
| 이미지 백업 | `/srv/vla/results/kw-vip/images/kw-vip-lab_x86.tar` |

현재 `<run>`: `ik_gt_rules`, `ik_cam_rules`, `act_rules`. jspark 계정에서 직접 실행한 결과는 저장소의 `outputs/` 에 있습니다.

### 6.2 확인 방법

```bash
# 비교표
vla-shell-kwvip.sh bash -c "python eval/summarize.py /results/kw-vip/eval/*/trials.jsonl"
# 실패한 시도만 보기
grep '"success": false' /srv/vla/results/kw-vip/eval/act_rules/trials.jsonl
# 영상을 내 컴퓨터로 가져오기 (서버에는 화면이 없음)
scp <서버계정>@<서버주소>:/srv/vla/results/kw-vip/eval/act_rules/videos/act_rules_20000.mp4 .
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
> git 작업(commit, push)은 **jspark 계정과 각자 노트북에서만** 합니다. vipuser 는 실행만 합니다.

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

서버의 `/home/jspark/kw-vip` 가 vipuser 실행에 쓰이는 코드입니다. PR 이 합쳐지면 jspark 가:
```bash
cd /home/jspark/kw-vip && git switch main && git pull
cp scripts/vla-shell-kwvip.sh /tmp/      # 스크립트가 바뀐 경우만, 이후 vipuser 가 ~/bin 에 다시 복사
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
| `vla-shell` 실행 시 `pull access denied for vla-lab` | 이미지가 서버에서 지워짐 | `vla-shell-kwvip.sh` 사용. 근본 해결은 관리자에게 이미지 복구와 Coolify 정리 제외 요청 |
| 컨테이너 안에서 코드가 안 보임 | vla-shell 은 `/srv/vla/workspace` 만 연결 | `vla-shell-kwvip.sh` 사용, `KWVIP_REPO` 확인 |
| `[kw-vip] .venv 가 없습니다` | 저장소에 venv 없음 | jspark 로 `scripts/setup_venv.sh act` |
| 해석이 항상 `rules(fallback)` | Ollama 미설치 또는 접속 불가 | `bash scripts/check_env.sh` 로 확인, Ollama 설치 후 `ollama pull <모델>` |
| 영상 글자가 □ 로 나옴 | 한글 폰트 없음 | `assets/fonts/NanumGothic-Regular.ttf` 가 있는지 확인, 또는 `KWVIP_FONT=<폰트경로>` |
| 종료 시 `EGLError` 경고가 잔뜩 출력 | MuJoCo 렌더러 정리 시 경고 | 결과에 영향 없음, 무시 |
| `lerobot-train: unrecognized arguments` | LeRobot 버전마다 옵션 이름이 다름 | `lerobot-train --help` 로 확인 (0.7.0 은 `--wandb` 대신 `--trackio.enable`) |
| ACT 가 변화 조건에서 실패 | 학습 범위 밖 일반화 격차 | 예상된 결과. 발표의 분석 재료로 사용 |
