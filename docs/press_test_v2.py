# 사용법: MUJOCO_GL=egl python press_test_v2.py models/scene_v2.xml   (버튼 3개 x 패널 이동 7가지를 IK+ctrl로 실제 눌러 봄)
import numpy as np, mujoco, itertools
import sys
XML = sys.argv[1] if len(sys.argv) > 1 else "models/scene_v2.xml"
m = mujoco.MjModel.from_xml_path(XML)
LO, HI = m.jnt_range[:6,0], m.jnt_range[:6,1]
TIP = m.site("press_tip").id; L6 = m.body("link6").id
PANEL = m.body("elevator_panel").id; P0 = m.body_pos[PANEL].copy()
SL = {i: m.joint(f"button_{i}_slide").qposadr[0] for i in (1,2,3)}
def ik(d, target, zdir, q0, iters=400, lam=0.05):
    d.qpos[:6] = q0
    for _ in range(iters):
        mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
        R = d.xmat[L6].reshape(3,3); p = d.site_xpos[TIP]
        jp, jr = np.zeros((3,m.nv)), np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,jr,TIP)
        e = np.r_[target-p, .5*np.cross(R[:,2], zdir)]; J = np.r_[jp[:,:6], .5*jr[:,:6]]
        d.qpos[:6] = np.clip(d.qpos[:6] + J.T@np.linalg.solve(J@J.T+lam**2*np.eye(6), e), LO, HI)
    mujoco.mj_kinematics(m,d); R = d.xmat[L6].reshape(3,3)
    return d.qpos[:6].copy(), np.linalg.norm(target-d.site_xpos[TIP]), np.degrees(np.arccos(np.clip(R[:,2]@zdir,-1,1)))
def plan(face, into, rng):
    dk = mujoco.MjData(m)
    best = min((ik(dk, face-0.05*into, into, rng.uniform(LO,HI)) for _ in range(30)), key=lambda s: s[1]+s[2]/1000)
    if best[1] > .003 or best[2] > 3: return None
    w = [best[0]]
    for dep in np.linspace(0,.016,9)[1:]:
        q, e, a = ik(dk, face+dep*into, into, w[-1], iters=200)
        if e > .004: return None
        w.append(q)
    return w
def trial(floor, off, seed=0):
    m.body_pos[PANEL] = P0 + off
    d = mujoco.MjData(m); mujoco.mj_forward(m,d)
    face = d.geom(f"button_{floor}_geom").xpos + [0,.015,0]; into = np.array([0,-1.,0])
    w = plan(face, into, np.random.default_rng(seed))
    if w is None: return "IK_FAIL", 0, 0
    d.ctrl[:6] = d.qpos[:6]; d.ctrl[6] = 0; peak = {i:0 for i in SL}
    def move(q, T):
        q0 = d.ctrl[:6].copy(); n = int(T/m.opt.timestep)
        for k in range(n):
            s=(k+1)/n; s=3*s*s-2*s**3; d.ctrl[:6]=(1-s)*q0+s*q; mujoco.mj_step(m,d)
            for i in SL: peak[i]=max(peak[i], d.qpos[SL[i]])
    move(w[0],3.0)
    for q in w[1:]: move(q,.4)
    move(w[0],1.0)
    tgt = peak[floor]*1000; other = max(peak[i] for i in SL if i!=floor)*1000
    return ("OK" if tgt>=10 and other<3 else "FAIL"), tgt, other
offs = {"base": [0,0,0], "z+5cm":[0,0,.05], "z-5cm":[0,0,-.05], "x+3cm":[.03,0,0], "x-3cm":[-.03,0,0], "y-4cm(farther)":[0,-.04,0], "y+3cm(closer)":[0,.03,0]}
for name, off in offs.items():
    res = [trial(f, np.array(off)) for f in (1,2,3)]
    print(f"{name:15s} " + "  ".join(f"F{f}:{r[0]} {r[1]:.1f}/{r[2]:.1f}mm" for f, r in zip((1,2,3), res)))
