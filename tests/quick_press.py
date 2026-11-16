import sys, time; sys.path.insert(0, ".")
from sim.scene import make_scene
from control.press_fsm import execute
for cond, f, s in [("none",1,0),("none",2,0),("none",3,0),("train",None,1),("variation",None,20000)]:
    sim = make_scene(s, cond, floor=f); t=time.time()
    r = execute({"intent":"press","floor":sim.floor}, sim)
    print(cond, sim.floor, sim.offset_kind, sim.panel_offset.round(3), r["success"], r["peak_mm"], r["other_mm"], r["fail_type"], r["t_task"], f"{time.time()-t:.1f}s")
