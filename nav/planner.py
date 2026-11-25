"""A* (8-연결) + 장애물 근접 비용 + 직선화(line-of-sight).

비용 = 이동 거리 + W_NEAR * max(0, SOFT - clearance)  -> 장애물에서 되도록 떨어진 경로.
시작 칸이 팽창 영역 안이면(장애물 옆에서 출발) 출발 주변 0.6m 안에서 '더 깊이 들어가지 않는' 칸을 허용해 빠져나온다.
"""
from __future__ import annotations

import heapq

import numpy as np

from nav.grid import ROBOT_R, OccupancyGrid

SOFT = 1.0
ESCAPE_R = 0.6
W_NEAR = 3.0
NB = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
      (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]


def astar(grid: OccupancyGrid, start, goal, r: float = ROBOT_R):
    clr = grid.clearance()
    s, g = grid.cell(start), grid.cell(goal)
    if clr[g] < r:
        return None
    free = clr >= r
    if clr[s] < r:
        # 출발점이 팽창 영역 안(장애물 옆으로 붙어 지나간 뒤 등): 출발 주변 0.6m 안에서는 출발점보다 더 깊이 들어가지
        # 않는 칸을 허용 -> 빠져나오는 경로가 생긴다 (그 밖은 원래 반경 그대로)
        ii, jj = np.meshgrid(np.arange(grid.nx), np.arange(grid.ny), indexing="ij")
        near = np.hypot(ii - s[0], jj - s[1]) * grid.res < ESCAPE_R
        free |= near & (clr >= clr[s] - 0.02)
    pen = W_NEAR * np.maximum(0.0, SOFT - clr)
    gx, gy = g
    openq = [(0.0, 0.0, s)]
    best = {s: 0.0}
    parent = {s: None}
    while openq:
        _, c, u = heapq.heappop(openq)
        if u == g:
            break
        if c > best.get(u, np.inf):
            continue
        for di, dj, w in NB:
            v = (u[0] + di, u[1] + dj)
            if not (0 <= v[0] < grid.nx and 0 <= v[1] < grid.ny) or not free[v]:
                continue
            nc = c + w * grid.res * (1.0 + pen[v])
            if nc < best.get(v, np.inf):
                best[v] = nc
                parent[v] = u
                h = np.hypot(v[0] - gx, v[1] - gy) * grid.res
                heapq.heappush(openq, (nc + h, nc, v))
    if g not in parent:
        return None
    cells = [g]
    while parent[cells[-1]] is not None:
        cells.append(parent[cells[-1]])
    pts = [grid.center(c) for c in reversed(cells)]
    if len(pts) < 2:
        return [np.asarray(start, float)[:2], np.asarray(goal, float)[:2]]
    pts[0], pts[-1] = np.asarray(start, float)[:2], np.asarray(goal, float)[:2]
    return simplify(grid, pts, r)


def simplify(grid, pts, r=ROBOT_R):
    """앞에서부터 '직선으로 갈 수 있는 가장 먼 점'만 남긴다."""
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not grid.segment_free(pts[i], pts[j], r * 0.98):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def path_length(path) -> float:
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))
