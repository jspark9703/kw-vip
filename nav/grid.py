"""점유격자 (5cm).  라이다 점을 누적하고, 거리변환으로 '가장 가까운 장애물까지 거리(clearance)'를 만든다.
휠체어는 원(반경 ROBOT_R)으로 근사: 기준점(베이스 원점)에서 가장 먼 몸체 끝이 ~0.65m 이므로 0.68m.
clearance >= ROBOT_R 인 칸만 지나갈 수 있고, 그 안에서는 제자리 회전도 안전하다.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

RES = 0.05
BOUNDS = (-1.0, 8.0, -0.6, 4.2)   # xmin, xmax, ymin, ymax
ROBOT_R = 0.68


class OccupancyGrid:
    def __init__(self, bounds=BOUNDS, res=RES):
        self.x0, self.x1, self.y0, self.y1 = bounds
        self.res = res
        self.nx = int(round((self.x1 - self.x0) / res))
        self.ny = int(round((self.y1 - self.y0) / res))
        self.hits = np.zeros((self.nx, self.ny), np.int32)
        self._clr = None

    def cell(self, p) -> tuple[int, int]:
        i = int(np.clip((p[0] - self.x0) / self.res, 0, self.nx - 1))
        j = int(np.clip((p[1] - self.y0) / self.res, 0, self.ny - 1))
        return i, j

    def center(self, ij) -> np.ndarray:
        return np.array([self.x0 + (ij[0] + 0.5) * self.res, self.y0 + (ij[1] + 0.5) * self.res])

    def add_points(self, pts: np.ndarray) -> int:
        """새로 점유된 칸 수를 반환."""
        if len(pts) == 0:
            return 0
        i = np.clip(((pts[:, 0] - self.x0) / self.res).astype(int), 0, self.nx - 1)
        j = np.clip(((pts[:, 1] - self.y0) / self.res).astype(int), 0, self.ny - 1)
        new = int(np.count_nonzero(self.hits[i, j] == 0))
        np.add.at(self.hits, (i, j), 1)
        if new:
            self._clr = None
        return new

    @property
    def occ(self) -> np.ndarray:
        return self.hits > 0

    def clearance(self) -> np.ndarray:
        """각 칸에서 가장 가까운 점유 칸까지 거리 (m)."""
        if self._clr is None:
            occ = self.occ
            self._clr = ndimage.distance_transform_edt(~occ) * self.res if occ.any() else np.full(occ.shape, 99.0)
        return self._clr

    def clearance_at(self, p) -> float:
        return float(self.clearance()[self.cell(p)])

    def segment_free(self, a, b, r=ROBOT_R) -> bool:
        clr = self.clearance()
        n = max(2, int(np.linalg.norm(np.asarray(b) - a) / (self.res * 0.5)))
        for t in np.linspace(0, 1, n):
            if clr[self.cell(a + t * (np.asarray(b) - a))] < r:
                return False
        return True


def gt_grid(model, data) -> OccupancyGrid:
    """정답 격자 (장면 생성 시 실행 가능성 검사용): 벽·장애물 geom 을 그대로 그린다(축 정렬 박스·원기둥)."""
    g = OccupancyGrid()
    xs = g.x0 + (np.arange(g.nx) + 0.5) * g.res
    ys = g.y0 + (np.arange(g.ny) + 0.5) * g.res
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    for k in range(model.ngeom):
        if not model.geom_contype[k] & 4:
            continue
        c = data.geom_xpos[k]
        s = model.geom_size[k]
        if model.geom_type[k] == 6:  # box
            m = (np.abs(X - c[0]) <= s[0]) & (np.abs(Y - c[1]) <= s[1])
        else:  # cylinder
            m = (X - c[0]) ** 2 + (Y - c[1]) ** 2 <= s[0] ** 2
        g.hits[m] += 1
    g._clr = None
    return g
