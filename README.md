# kw-vip — 텍스트 명령으로 동작하는 생활보조 로봇 (시나리오 A: 엘리베이터 버튼)

**v4 (최종 태스크)**: "현관문 열어줘" 같은 한국어 명령을 해석해, 전동 휠체어가 아파트 복도를 **주행**(라이다로 택배·사람 우회)하고
우리 집(1203호) 현관 도어락 앞에 **도킹**한 뒤, 3x4 키패드의 **글자**로 키 위치를 찾아 **미리 등록된 비밀번호 + #** 을 키마다 피드백 제어로 누르고,
**상태 LED**(초록 = 열림)로 문이 열렸는지 판단한다. 설계·모델 변경점·점검 결과: [docs/kw-vip_v4_설계.md](docs/kw-vip_v4_설계.md)
(v3 엘리베이터 버튼 태스크는 v4 로 교체했다.)

**v2 (이전 단계, `--scene v2`)**: 고정 휠체어에서 색 버튼 3개(1·2·3층)를 누른다. IK·카메라·ACT 비교는 이 장면에서 했다.

팀원용 사용 설명서(실행·학습/평가·산출물·Git/PR): [docs/kw-vip_개발가이드.md](docs/kw-vip_개발가이드.md)
v2 설계와 단계별 통과 조건은 [docs/kw-vip과제_구현개발프로세스.md](docs/kw-vip과제_구현개발프로세스.md) 참고.

```
v4
텍스트 ─ nlu (규칙 안전장치 + 로컬 LLM, nlu/doorlock.py) ─> {"intent": unlock|wait|reject|clarify}
  └ mission/fsm.py : STOW_ARM → NAV_TO_LOBBY → FIND_PANEL → DOCK → OBSERVE/READ_KEYS → (키마다) APPROACH → PRESS → ... '#' → VERIFY → STOW_ARM
      nav/         라이다 점유격자 · A* · pure pursuit · 재계획 · 비상정지 · 도킹
      perception/  panel.py (head depth: 벽 RANSAC → 튀어나온 도어락, 사전정보로 이웃집 배제) · digits.py (wrist RGB-D 정면화 → 키 글자 NCC, LED 색)
      control/     observe.py (관찰 자세) · press_feedback.py (키마다 1mm 서보 + 변위·접촉력 피드백, 재시도, * 재입력)
판정 ─ sim/world.py judge(): 도어락 열림(마지막 * 이후 입력 = 비밀번호+#), 휠체어-벽/장애물 충돌 0

v2
텍스트 ─ nlu ─> 목표 위치 (site_gt | camera 색) ─> control/press_fsm.py (IK) | policy/act_runner.py (ACT) ─> sim/scene.py judge()
```

## 폴더

| 경로 | 내용 |
|---|---|
| `models/` | v4 `scene_v4.xml` ← `make_scene_v4.py`(주행 관절·스타일러스·아파트 복도·도어락·라이다·택배/사람), v2 `scene_v2.xml` ← `make_scene_v2.py`, 원본 `orig/` |
| `assets/` | 한글 폰트(OFL), `textures/`(키 라벨 `key_*.png`·호수판·소화전 표지, make_scene_v4.py 가 생성) |
| `sim/` | `world.py` make_world(seed, condition) (v4: 도어락 동작·충돌 집계), `scene.py` make_scene (v2), `recorder.py` 오버레이 mp4 (v4: 추종 카메라·미니맵·입력 상태) |
| `nav/` | `lidar.py`, `grid.py`, `planner.py`, `follower.py`, `dock.py` (주행·우회·도킹) |
| `mission/` | `fsm.py` v4 상위 상태 기계 run_mission(task, sim) |
| `control/` | `ik.py` press_tip DLS IK, `press_fsm.py` execute(task, sim) (v2), `observe.py`·`press_feedback.py` (v4) |
| `nlu/` | `schema.py`, `rules.py`·`validate.py`(v2), `doorlock.py`(v4), `llm_ollama.py`, `parse.py`, `testset_ko.yaml`(v2 40문장), `testset_v4_ko.yaml`(v4 24문장) |
| `perception/` | `site_gt.py`, `camera.py`(v2 색), `panel.py`(v4 도어락), `digits.py`(v4 키 글자·LED) |
| `policy/` | `gen_demos.py`(IK 전문가 데모), `to_lerobot.py`, `train_act.sh`, `act_runner.py` |
| `eval/` | `eval_config.yaml`(v2), `eval_config_v4.yaml`(v4), `run_eval.py --scene`, `summarize.py`(v4 서브태스크 표 포함) |
| `pipeline.py` / `run_demo.py` | 한 시도 실행 / 텍스트 입력 데모 |
| `tests/` | v4: `check_scene_v4.py`, `check_nav.py [--camera]`, `check_keypad.py`, `check_unlock.py` / v2: `phase_checks.py`, `check_camera.py` / 공통: `eval_nlu.py [--scene v4]` |
| `scripts/` | `vla-shell-kwvip.sh`(서버 실행 환경), `setup_venv.sh`, `check_env.sh` |
| `docker/Dockerfile` | vla-lab 이미지 대체용 경량 런타임 |
| `outputs/` | 로그·영상·데이터셋·체크포인트 (git 제외) |

## 실행

```bash
# v4 (기본): 복도 주행 + 현관 도어락.  영상: outputs/videos/demo_<seed>_1.mp4 (추종 카메라 + 미니맵 + 입력 상태)
MUJOCO_GL=egl python run_demo.py "현관문 열어줘" --parser rules
MUJOCO_GL=egl python run_demo.py "집에 들어가자" --condition variation --seed 7    # 도어락 위치 변화 + 택배·사람 2~3개
MUJOCO_GL=egl python run_demo.py "옆집 문 열어줘"                                # 움직이지 않고 거절
#   --runner ik_gt : 도킹 목표·키 위치를 시뮬레이터 좌표로 (기준선)   --password 2580 : 등록 비밀번호 지정

# v4 단계별 점검
MUJOCO_GL=egl python tests/check_scene_v4.py
MUJOCO_GL=egl python tests/check_nav.py --camera --n 20
MUJOCO_GL=egl python tests/check_keypad.py --n 30
MUJOCO_GL=egl python tests/check_unlock.py --n 10 [--inject-miss 0.010]
python tests/eval_nlu.py --scene v4 -v

# v4 고정 시드 평가 (기본 10 + 변화 10 + 명령 6) -> outputs/eval/v4_<runner>_<parser>/
MUJOCO_GL=egl python eval/run_eval.py --scene v4 --runner ik_cam --parser rules
python eval/summarize.py outputs/eval/v4_*/trials.jsonl

# ---- v2 (이전 장면) ----
# 텍스트 명령 1회 (영상: outputs/videos/demo_<seed>_1.mp4)
MUJOCO_GL=egl python run_demo.py "엘리베이터 3층을 선택해 줘" --scene v2 --seed 3 --condition variation
MUJOCO_GL=egl python run_demo.py "버튼은 누르지 말고 대기해" --scene v2
MUJOCO_GL=egl python run_demo.py                      # 대화형 (모호하면 되물음, v4)
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

## v4 실측 결과 (2026-11-24, kw-vip-lab 컨테이너, 파서 rules)

| 실행부 | 기본 | 변화 | 성공 시 시간(중앙값) | 명령 | 키 판독 | 충돌 | 키당 누름 | LED=센서 |
|---|---|---|---|---|---|---|---|---|
| ik_cam (센서만, 최종) | 10/10 | 10/10 | 57.7s | 6/6 | 240/240 | 0 | 1.00 | 20/20 |
| ik_gt (GT 기준선) | 10/10 | 10/10 | 51.1s | 6/6 | 240/240 | 0 | 1.01 | 20/20 |

추가 반복(독립 시드): variation 30회 30/30. 평가 범위 밖 `stress`(도어락 높이 0.92~1.32m, 장애물 3개) 20회 15/20 —
실패 5회 모두 도어락 높이가 팔 도달 범위(1.00~1.22m) 밖 (ik 4, digit 1). 대표 영상 `outputs/videos/v4/`.
서브태스크 표·실패 분석: [docs/kw-vip_v4_설계.md](docs/kw-vip_v4_설계.md) 6장.

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
