"""텍스트 명령 -> 해석 -> 검증 -> 실행 -> 결과 (+ 오버레이 mp4).

예)
  MUJOCO_GL=egl python run_demo.py "현관문 열어줘"                               # scene v4 (기본): 복도 주행 + 도어락
  MUJOCO_GL=egl python run_demo.py "집에 들어가자" --condition variation --seed 7  # 도어락 위치 변화 + 택배·사람 2~3개
  MUJOCO_GL=egl python run_demo.py "2층 눌러줘" --scene v2                       # 기존 고정 휠체어 3버튼
  MUJOCO_GL=egl python run_demo.py "엘리베이터 3층을 선택해 줘" --seed 3 --condition variation --runner ik_cam
  MUJOCO_GL=egl python run_demo.py            # 대화형: 명령을 계속 입력, 모호하면 되물음
옵션 --parser rules 를 주면 LLM(Ollama) 없이 규칙만 쓴다. Ollama 가 없으면 자동으로 규칙 폴백.
"""
import argparse
import re
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from nlu.parse import parse_command  # noqa: E402
from pipeline import OUT, RUNNERS, close_all, run_episode  # noqa: E402


def show(r):
    p = r["parsed"]
    llm = ""
    if r["parser"] == "rules+llm":
        llm = f", LLM {r['llm_ms']:.0f}ms"
    elif r["parser"] == "rules(fallback)":
        llm = f", LLM 실패 -> 규칙 사용: {(r.get('llm_error') or '')[:60]}"
    print(f"  해석[{r['parser']}{llm}]: "
          f"intent={p['intent']} floor={p['floor']} ({p['reason']})")
    if p["intent"] == "press":
        print(f"  결과: {'성공' if r['success'] else '실패'}  목표 {r['peak_mm']}mm / 다른 버튼 {r['other_mm']}mm"
              f"  fail_type={r['fail_type']}  작업시간 {r['t_task_s']}s")
    elif p["intent"] == "unlock":
        print(f"  결과: {'문 열림' if r['success'] else '실패'}  비밀번호 {r.get('password_len')}자리 + #, "
              f"입력 {len(r.get('entered') or '')}회, 틀린 입력 {r.get('wrong_inputs')}  fail_type={r['fail_type']}  작업시간 {r['t_task_s']}s")
        d, n, k = r.get("dock") or {}, r.get("nav") or {}, r.get("keys") or {}
        print(f"  주행 {n.get('traveled_m')}m 재계획 {n.get('replans')} / 충돌 {r.get('collisions')} / 도킹 오차 {d.get('dock_err_gt_mm')}mm"
              f" / 키 판독 {k.get('correct')}/{k.get('n_keys')} / 누르기 {(r.get('press') or {}).get('results')}"
              f" / LED {((r.get('press') or {}).get('verify') or {}).get('led')}")
    else:
        print(f"  로봇은 움직이지 않음 -> {p['reason']}  (키·버튼 최대 변위 {r['other_mm']}mm)")
    if r.get("message"):
        print(f"  로봇: {r['message']}")
    if r["video"]:
        print(f"  영상: {r['video']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", help="텍스트 명령 (없으면 대화형)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="base", choices=["none", "train", "base", "variation", "stress"])
    ap.add_argument("--parser", default="rules+llm", choices=["rules", "rules+llm"])
    ap.add_argument("--scene", default="v4", choices=["v2", "v4"], help="v4: 복도 주행 + 현관 도어락 / v2: 고정 휠체어 3버튼")
    ap.add_argument("--runner", default=None, choices=RUNNERS, help="기본: v4 는 ik_cam, v2 는 ik_gt")
    ap.add_argument("--password", default=None, help="v4 등록 비밀번호 (기본: 시드가 정한 값, condition none 은 2580)")
    ap.add_argument("--act-ckpt", default=os.environ.get("KWVIP_ACT_CKPT"), help="runner=act 일 때 체크포인트 경로")
    ap.add_argument("--video", default=None, help="mp4 경로 (기본 outputs/videos/demo_<seed>_<n>.mp4)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--json", action="store_true", help="결과 JSON 출력")
    a = ap.parse_args()
    a.runner = a.runner or ("ik_cam" if a.scene == "v4" else "ik_gt")
    if a.scene == "v4" and a.runner == "act":
        ap.error("ACT 정책은 scene v2 전용입니다")
    from sim.scene import FLOORS
    floors = FLOORS

    policy = None
    if a.runner == "act":
        from policy.act_runner import load_policy
        policy = load_policy(a.act_ckpt)

    n = 0

    def once(text, task_override=None):
        nonlocal n
        n += 1
        video = None if a.no_video else (a.video if (a.video and a.command) else OUT / "videos" / f"demo_{a.seed}_{n}.mp4")
        r = run_episode(text, a.seed, a.condition, parser=a.parser, runner=a.runner, video=video,
                        act_policy=policy, task_override=task_override, run_name="demo", scene=a.scene,
                        password=a.password)
        show(r)
        if a.json:
            print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
        return r

    if a.command:
        once(a.command)
    else:
        print("명령을 입력하세요 (종료: q).  예: 현관문 열어줘 / 문 열지 말고 기다려 / 옆집 문 열어줘"
              if a.scene == "v4" else "명령을 입력하세요 (종료: q).  예: 2층 눌러줘 / 버튼은 누르지 말고 대기해 / 7층 눌러줘")
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if text in ("q", "quit", "exit", "종료"):
                break
            if not text:
                continue
            task, meta = parse_command(text, a.parser, scene_floors=floors, scene=a.scene)
            if task["intent"] == "clarify":  # 되묻기: 모호할 때만
                print(f"  로봇: {task['reason']}")
                ans = input("  (답) > ").strip()
                task2, _ = parse_command(ans, a.parser, scene_floors=floors, scene=a.scene)
                if a.scene == "v4" and re.match(r"^\s*(네|예|응|그래|ㅇㅇ|좋아|yes|ok)", ans, re.I):
                    task2 = {"intent": "unlock", "floor": None, "reason": "되묻기에 동의"}
                if task2["intent"] in ("press", "unlock"):
                    once(f"{text} -> {ans}", task_override=task2)
                    continue
                print(f"  로봇: 이해하지 못했습니다 ({task2['intent']}: {task2['reason']}). 움직이지 않습니다.")
                continue
            once(text)
    close_all()


if __name__ == "__main__":
    main()
