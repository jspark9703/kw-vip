"""Ollama 없이 LLM 경로 검증: 가짜 /api/chat 서버로 parse_command('rules+llm') 의 병합·검증·폴백을 확인.
사용법: python tests/test_nlu_llm_mock.py
"""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ANSWERS = {  # 가짜 LLM 답 (사용자 문장 -> JSON 문자열)
    "맨 위층 버튼 눌러줘": {"intent": "press", "floor": 3, "reason": "꼭대기층"},
    "로비로 가야 하니까 버튼 좀 눌러줘": {"intent": "press", "floor": 1, "reason": "로비"},
    "2층 눌러줘": {"intent": "press", "floor": 3, "reason": "오답"},          # 규칙(2층)과 충돌 -> clarify
    "지하 2층으로": {"intent": "press", "floor": -2, "reason": "지하"},        # 장면에 없음 -> reject
    "아무거나": "not json",                                                  # 깨진 출력 -> 규칙 폴백
}
seen = {}

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["messages"][-1]["content"]
        seen["format"] = body.get("format"); seen["temperature"] = body["options"]["temperature"]
        ans = ANSWERS.get(text, {"intent": "reject", "floor": None, "reason": "?"})
        content = ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False)
        out = json.dumps({"message": {"role": "assistant", "content": content}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)
    def log_message(self, *a): pass

srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ["OLLAMA_HOST"] = f"http://127.0.0.1:{srv.server_port}"
from nlu.parse import parse_command

cases = [("맨 위층 버튼 눌러줘", "press", 3, "rules+llm"),
         ("로비로 가야 하니까 버튼 좀 눌러줘", "press", 1, "rules+llm"),
         ("2층 눌러줘", "clarify", None, "rules+llm"),
         ("지하 2층으로", "reject", None, "rules+llm"),
         ("아무거나", "reject", None, "rules(fallback)"),
         ("1층 버튼은 누르지 마", "wait", None, "rules(block)")]   # 규칙 block 이면 LLM 호출 안 함
ok = True
for text, intent, floor, parser in cases:
    task, meta = parse_command(text, "rules+llm")
    good = task["intent"] == intent and task["floor"] == floor and meta["parser"] == parser
    ok &= good
    print("OK " if good else "BAD", f"{text!r:32s} -> {task['intent']}/{task['floor']} [{meta['parser']}] {task['reason']}")
assert seen["format"]["required"] == ["intent", "floor", "reason"] and seen["temperature"] == 0
print("ALL PASS" if ok else "FAILED"); sys.exit(0 if ok else 1)
