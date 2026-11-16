# kw-vip — 텍스트 명령으로 동작하는 생활보조 로봇 (시나리오 A: 엘리베이터 버튼)

MuJoCo 휠체어–PiPER 모델에서 "2층 눌러줘" 같은 한국어 명령을 해석해 버튼 3개(1·2·3층) 중 하나를 실제 물리 접촉으로 누른다.

팀원용 사용 설명서(실행·학습/평가·산출물·Git/PR): [docs/kw-vip_개발가이드.md](docs/kw-vip_개발가이드.md)
설계와 단계별 통과 조건은 [docs/kw-vip과제_구현개발프로세스.md](docs/kw-vip과제_구현개발프로세스.md) 참고.

```
텍스트 ─ nlu/rules.py (부정·대기 안전장치, 층수 추출)
       └ nlu/llm_ollama.py (Ollama 로컬 LLM, JSON 출력) ─ nlu/validate.py ─> {"intent","floor","reason"}
목표 위치 ─ perception/site_gt.py (시뮬레이터 좌표, 기본) | perception/camera.py (head_rgb RGB+depth)
실행 ─ control/press_fsm.py (IK 상태 기계, 기준선) | policy/act_runner.py (ACT 정책)
판정 ─ sim/scene.py judge(): 목표 버튼 slide 변위 ≥ 10mm 이고 다른 버튼 < 3mm
```

## 폴더

| 경로 | 내용 |
|---|---|
| `models/` | `scene_v2.xml`(버튼 3개·press_tip·큐브 축소·place_zone), 생성 스크립트 `make_scene_v2.py`, 원본 `orig/` |
| `sim/` | `scene.py` make_scene(seed, condition), `recorder.py` 오버레이 mp4 |
| `control/` | `ik.py` press_tip DLS IK, `press_fsm.py` execute(task, sim) |
| `nlu/` | `schema.py`, `rules.py`, `llm_ollama.py`, `validate.py`, `parse.py`, `testset_ko.yaml`(40문장 초안) |
| `perception/` | `site_gt.py`, `camera.py` |
| `policy/` | `gen_demos.py`(IK 전문가 데모), `to_lerobot.py`, `train_act.sh`, `act_runner.py` |
| `eval/` | `eval_config.yaml`(고정 시드), `run_eval.py`, `summarize.py` |
| `pipeline.py` / `run_demo.py` | 한 시도 실행 / 텍스트 입력 데모 |
| `tests/` | `phase_checks.py`(Phase 0·1 통과 조건), `eval_nlu.py`(Phase 2 표), `check_camera.py` |
| `scripts/` | `vla-shell-kwvip.sh`(서버 실행 환경), `setup_venv.sh`, `check_env.sh` |
| `docker/Dockerfile` | vla-lab 이미지 대체용 경량 런타임 |
| `outputs/` | 로그·영상·데이터셋·체크포인트 (git 제외) |

## 실행

```bash
# 텍스트 명령 1회 (영상: outputs/videos/demo_<seed>_1.mp4)
MUJOCO_GL=egl python run_demo.py "엘리베이터 3층을 선택해 줘" --seed 3 --condition variation
MUJOCO_GL=egl python run_demo.py "버튼은 누르지 말고 대기해"
MUJOCO_GL=egl python run_demo.py                      # 대화형 (모호하면 되물음)
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

## 서버 실측 결과 (2026-11-16, kw-vip-lab 컨테이너, RTX 4070 Ti)

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
