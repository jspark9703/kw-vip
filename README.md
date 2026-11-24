# kw-vip — 텍스트 명령으로 동작하는 생활보조 로봇 (시나리오 A: 엘리베이터 버튼)

**v3 (최종 태스크)**: "엘리베이터로 가서 3층 눌러줘" 같은 한국어 명령을 해석해, 전동 휠체어가 엘리베이터까지 **주행**(라이다로 장애물 우회)하고
패널 앞에 **도킹**한 뒤, 같은 색 버튼 6개(2x3) 중 **숫자 글씨**로 목표 버튼을 찾아 **피드백 제어로 누르고 램프·변위로 눌렸는지 판단**한다.
설계·모델 변경점·점검 결과: [docs/kw-vip_v3_설계.md](docs/kw-vip_v3_설계.md)

**v2 (이전 단계, `--scene v2`)**: 고정 휠체어에서 색 버튼 3개(1·2·3층)를 누른다. IK·카메라·ACT 비교는 이 장면에서 했다.

팀원용 사용 설명서(실행·학습/평가·산출물·Git/PR): [docs/kw-vip_개발가이드.md](docs/kw-vip_개발가이드.md)
v2 설계와 단계별 통과 조건은 [docs/kw-vip과제_구현개발프로세스.md](docs/kw-vip과제_구현개발프로세스.md) 참고.

```
v3
텍스트 ─ nlu (규칙 안전장치 + 로컬 LLM, 건물 1~6층, 색 매핑 없음) ─> {"intent","floor","reason"}
  └ mission/fsm.py : STOW_ARM → NAV_TO_LOBBY → FIND_PANEL → DOCK → OBSERVE/READ_DIGITS → PRESS_SERVO → VERIFY → STOW_ARM
      nav/         라이다 점유격자 · A* · pure pursuit · 재계획 · 비상정지 · 도킹
      perception/  panel.py (head depth: 벽 RANSAC → 튀어나온 패널) · digits.py (wrist RGB-D 정면화 → 숫자 템플릿 NCC)
      control/     observe.py (관찰 자세) · press_feedback.py (2mm 서보 + 변위·접촉력 피드백 + 재시도)
판정 ─ sim/world.py judge(): 목표 버튼 ≥10mm, 다른 버튼 <3mm, 휠체어-벽/장애물 충돌 0

v2
텍스트 ─ nlu ─> 목표 위치 (site_gt | camera 색) ─> control/press_fsm.py (IK) | policy/act_runner.py (ACT) ─> sim/scene.py judge()
```

## 폴더

| 경로 | 내용 |
|---|---|
| `models/` | v3 `scene_v3.xml` ← `make_scene_v3.py`(주행 관절·홀·숫자 버튼 6개·라이다·장애물), v2 `scene_v2.xml` ← `make_scene_v2.py`, 원본 `orig/` |
| `assets/` | 한글 폰트(OFL), `textures/digit_*.png`(버튼 숫자 라벨, make_scene_v3.py 가 생성) |
| `sim/` | `world.py` make_world(seed, condition) (v3), `scene.py` make_scene (v2), `recorder.py` 오버레이 mp4 (v3: 추종 카메라·미니맵) |
| `nav/` | `lidar.py`, `grid.py`, `planner.py`, `follower.py`, `dock.py` (v3 주행·우회·도킹) |
| `mission/` | `fsm.py` v3 상위 상태 기계 run_mission(task, sim) |
| `control/` | `ik.py` press_tip DLS IK, `press_fsm.py` execute(task, sim) (v2), `observe.py`·`press_feedback.py` (v3) |
| `nlu/` | `schema.py`, `rules.py`, `llm_ollama.py`, `validate.py`, `parse.py`, `testset_ko.yaml`(v2 40문장), `testset_v3_ko.yaml`(v3 24문장) |
| `perception/` | `site_gt.py`, `camera.py`(v2 색), `panel.py`(v3 패널), `digits.py`(v3 숫자) |
| `policy/` | `gen_demos.py`(IK 전문가 데모), `to_lerobot.py`, `train_act.sh`, `act_runner.py` |
| `eval/` | `eval_config.yaml`(v2), `eval_config_v3.yaml`(v3), `run_eval.py --scene`, `summarize.py`(v3 서브태스크 표 포함) |
| `pipeline.py` / `run_demo.py` | 한 시도 실행 / 텍스트 입력 데모 |
| `tests/` | v3: `check_scene_v3.py`, `check_nav.py [--camera]`, `check_digits.py`, `check_press_v3.py` / v2: `phase_checks.py`, `check_camera.py` / 공통: `eval_nlu.py [--scene v3]` |
| `scripts/` | `vla-shell-kwvip.sh`(서버 실행 환경), `setup_venv.sh`, `check_env.sh` |
| `docker/Dockerfile` | vla-lab 이미지 대체용 경량 런타임 |
| `outputs/` | 로그·영상·데이터셋·체크포인트 (git 제외) |

## 실행

```bash
# v3 (기본): 주행 + 숫자 버튼.  영상: outputs/videos/demo_<seed>_1.mp4 (추종 카메라 + 미니맵 + 상태)
MUJOCO_GL=egl python run_demo.py "엘리베이터로 가서 3층 눌러줘" --parser rules
MUJOCO_GL=egl python run_demo.py "5층 눌러줘" --condition variation --seed 7       # 패널 ±0.6m + 장애물 2~3개
MUJOCO_GL=egl python run_demo.py "7층 눌러줘"                                     # 움직이지 않고 거절
#   --runner ik_gt : 도킹 목표·버튼 위치를 시뮬레이터 좌표로 (기준선)

# v3 단계별 점검
MUJOCO_GL=egl python tests/check_scene_v3.py
MUJOCO_GL=egl python tests/check_nav.py --camera --n 20
MUJOCO_GL=egl python tests/check_digits.py --n 50
MUJOCO_GL=egl python tests/check_press_v3.py --n 12
python tests/eval_nlu.py --scene v3 -v

# v3 고정 시드 평가 (기본 10 + 변화 10 + 명령 6) -> outputs/eval/v3_<runner>_<parser>/
MUJOCO_GL=egl python eval/run_eval.py --scene v3 --runner ik_cam --parser rules
python eval/summarize.py outputs/eval/v3_*/trials.jsonl

# ---- v2 (이전 장면) ----
# 텍스트 명령 1회 (영상: outputs/videos/demo_<seed>_1.mp4)
MUJOCO_GL=egl python run_demo.py "엘리베이터 3층을 선택해 줘" --scene v2 --seed 3 --condition variation
MUJOCO_GL=egl python run_demo.py "버튼은 누르지 말고 대기해" --scene v2
MUJOCO_GL=egl python run_demo.py                      # 대화형 (모호하면 되물음, v3)
#   --parser rules        LLM 없이 규칙만
#   --runner ik_cam       카메라 인식으로 목표 위치
#   --runner act --act-ckpt <체크포인트>

# 단계별 통과 조건
MUJOCO_GL=egl python tests/phase_checks.py            # phase0, phase1, 변화 범위 도달성
python tests/eval_nlu.py -v                           # 40문장: 규칙만 / 규칙+LLM

# 고정 시드 평가 (기본 10 + 변화 10 + 명령 5)  ->  outputs/eval/<run>/
MUJOCO_GL=egl python eval/run_eval.py --runner ik_gt  --parser rules+llm
MUJOCO_GL=egl python eval/run_eval.py --runner ik_cam --parser rules+llm
python eval/summarize.py outputs/eval/*/trials.jsonl  # 발표용 비교표 (마크다운)
```

## 서버 실행 (jspark 계정)

원래의 `vla-shell` 은 쓰지 않고 `scripts/vla-shell-kwvip.sh` 로 실행한다 (GPU + MuJoCo EGL 컨테이너 `kw-vip-lab:x86`, 패키지는 `.venv`, 결과는 `outputs/`).
```bash
scripts/setup_venv.sh act                                  # 처음 1회
scripts/vla-shell-kwvip.sh                                 # 대화형 셸
scripts/vla-shell-kwvip.sh python run_demo.py "2층 눌러줘"
```
자세한 내용은 [docs/kw-vip_개발가이드.md](docs/kw-vip_개발가이드.md) 2장.

## Ollama (명령 해석 LLM)

서버·노트북 모두 로컬 Ollama 를 쓴다(외부 API 없음). Ollama 가 없거나 응답이 깨지면 규칙 파서로 자동 폴백하고 로그에 `rules(fallback)` 로 남는다.
```bash
ollama pull qwen2.5:3b        # 후보: qwen2.5:1.5b / qwen2.5:3b / llama3.2:3b -> tests/eval_nlu.py 로 비교해서 고름
export KWVIP_LLM_MODEL=qwen2.5:3b OLLAMA_HOST=http://127.0.0.1:11434
python tests/eval_nlu.py -v
```

## 노트북 이식 (최종 추론)

`git clone` → `pip install -r requirements.txt` (ACT 쓰면 `requirements-act.txt`) → Ollama 모델·ACT 체크포인트 복사 →
네트워크를 끊고 `python run_demo.py`. 경로는 모두 저장소 기준 상대 경로이고 GPU 가 없으면 CPU 로 돈다.
Windows 에서는 `MUJOCO_GL` 을 지정하지 않는다(기본 glfw).

## ACT (Phase 4) 실행 순서

```bash
# 모두 scripts/vla-shell-kwvip.sh 안에서 (결과는 outputs/...)
python policy/gen_demos.py --per-floor 80 --seed-start 100 --out outputs/datasets/press_v1
python policy/to_lerobot.py --src outputs/datasets/press_v1 --dst outputs/lerobot/press_v1
bash policy/train_act.sh press_v1 20000 32
python tests/act_check.py outputs/train/act_press_v1_s20000/checkpoints/020000/pretrained_model --seeds 500 530   # 체크포인트 선택(개발 시드)
python eval/run_eval.py --runner act --parser rules --run act_rules --act-ckpt <선택한 체크포인트>
```

## v3 실측 결과 (2026-11-24, kw-vip-lab 컨테이너, 파서 rules)

| 실행부 | 기본 SR | 변화 SR | 전체 SR | 성공 시 시간(중앙값) | 명령 적합성 | 숫자 판독 | 충돌 | 도킹 오차 |
|---|---|---|---|---|---|---|---|---|
| ik_cam (센서만, 최종) | 100% | 100% | 100% | 40.0s | 6/6 | 120/120 | 0 | 10.7mm |
| ik_gt (GT 기준선) | 100% | 100% | 100% | 32.7s | 6/6 | 120/120 | 0 | 2.2mm |

평가 범위 밖 `stress`(패널 ±1.0m / ±10cm, 장애물 3개) 16회: 12/16. 실패는 팔 작업 영역 밖 높이(ik) 3, 도킹 경로 막힘(nav) 1.
추가 반복(독립 시드): variation 30회 30/30, stress 20회 17/20 (nav 2: 대기 지점 막힘, ik 1: 높이). 대표 영상 `outputs/videos/v3/`.
서브태스크 표·실패 영상: [docs/kw-vip_v3_설계.md](docs/kw-vip_v3_설계.md) 5장.

## v2 서버 실측 결과 (2026-11-16, kw-vip-lab 컨테이너, RTX 4070 Ti)

| 실행부 | 기본 SR | 변화 SR | 전체 SR | 일반화 격차 | 성공 시 시간(중앙값) | 명령 적합성(5회) | 주요 실패 유형 |
|---|---|---|---|---|---|---|---|
| IK + 시뮬레이터 좌표 (기준선) | 100% | 100% | 100% | 0%p | 7.5s | 5/5 | - |
| IK + 카메라 인식 (위치 오차 중앙값 1.27mm) | 100% | 100% | 100% | 0%p | 7.5s | 5/5 | - |
| ACT (240 에피소드, 20k step) | 90% | 10% | 50% | 80%p | 10.0s(고정 실행 시간) | 5/5 | timeout(버튼 미접촉) 8, contact 2 |

- 파서: 이번 측정은 `rules` (서버에 Ollama 미설치). NLU 40문장: 규칙만 36/40, 금지·불가능 문장 오동작 0.
- ACT: 데모 240개(층당 80, IK 전문가, 30% 관절 교란), 관측 head/wrist 120x160 + 상태 10, 행동 ctrl 7, 20Hz,
  학습 21.5분, 체크포인트 198MB, 백본 ResNet18(torchvision ImageNet 사전학습). 개발 시드 500~529 에서 29/30.
  변화 조건(패널 z±5~8cm, 4~8cm 멀어짐)에서 거의 실패 → 학습 범위(±2cm) 밖 일반화 격차. 개선 방향: 학습 범위 확대, 카메라로 목표 위치를 넣는 하이브리드.
- 추론 지연(ACT 1스텝) 은 발표 규정상 노트북에서 다시 측정한다.
