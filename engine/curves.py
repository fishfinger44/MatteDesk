"""Bezier curves - ONE module for every place that needs an editable curve (user decision 2026-11-09).
First user: the edge warp of the Cleanup keys (VOID drew the horizon / an edge in the wrong place -> src curve = where
VOID put it, dst curve = where it must be). Front-end editor: static/curves.js (same format).

Format (JSON, image pixels):
    {"type": "bezier", "closed": false,
     "pts": [{"p": [x, y], "in": [dx, dy], "out": [dx, dy], "cusp": false}, ...]}
`in` / `out` = tangent handles RELATIVE to p (Nuke-style); cusp = the two handles are independent (a sharp corner),
otherwise the editor keeps them collinear. Segment i = cubic (p_i, p_i + out_i, p_{i+1} + in_{i+1}, p_{i+1}).

Functions: from_points (Catmull-Rom through points, exact), fit (least-squares fit of a noisy polyline within `tol` px,
corners kept as cusps), sample (dense points + normalized arc length + segment parameter), correspond (pairs of points
on two curves: by arc length, by x, by y, or by segment parameter), warp_field / apply_warp (smooth backward map that
moves the src curve onto the dst curve, fading out over `reach` px)."""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------- format


def point(p, tin=(0.0, 0.0), tout=(0.0, 0.0), cusp=False):
    return {"p": [float(p[0]), float(p[1])], "in": [float(tin[0]), float(tin[1])],
            "out": [float(tout[0]), float(tout[1])], "cusp": bool(cusp)}


def spline(pts, closed=False):
    return {"type": "bezier", "closed": bool(closed), "pts": list(pts)}


def is_spline(obj) -> bool:
    return isinstance(obj, dict) and obj.get("type") == "bezier" and isinstance(obj.get("pts"), list)


def normalize(obj) -> dict:
    """Spline from a spline dict or a plain point list [[x, y], ...] (Catmull-Rom through the points)."""
    if is_spline(obj):
        out = []
        for q in obj["pts"]:
            out.append(point(q["p"], q.get("in", (0, 0)), q.get("out", (0, 0)), q.get("cusp", False)))
        return spline(out, obj.get("closed", False))
    return from_points(obj)


def from_points(pts, closed=False) -> dict:
    """Uniform Catmull-Rom through the points, written as Bezier handles (identical curve: handle = (p+1 - p-1) / 6)."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    n = len(p)
    out = []
    for i in range(n):
        if closed:
            a, b = p[(i - 1) % n], p[(i + 1) % n]
        else:
            a = p[i - 1] if i > 0 else 2 * p[0] - p[1] if n > 1 else p[0]
            b = p[i + 1] if i < n - 1 else 2 * p[-1] - p[-2] if n > 1 else p[-1]
        t = (b - a) / 6.0
        out.append(point(p[i], -t, t))
    return spline(out, closed)


def segments(sp) -> list:
    """List of (4, 2) control polygons."""
    sp = normalize(sp)
    P = sp["pts"]
    n = len(P)
    segs = []
    last = n if sp["closed"] and n > 2 else n - 1
    for i in range(last):
        a, b = P[i], P[(i + 1) % n]
        p0 = np.array(a["p"]); p3 = np.array(b["p"])
        segs.append(np.array([p0, p0 + a["out"], p3 + b["in"], p3]))
    return segs


def bez(c, t):
    t = np.asarray(t, np.float64)[..., None]
    mt = 1 - t
    return mt ** 3 * c[0] + 3 * mt * mt * t * c[1] + 3 * mt * t * t * c[2] + t ** 3 * c[3]


def bez_d1(c, t):
    t = np.asarray(t, np.float64)[..., None]
    mt = 1 - t
    return 3 * mt * mt * (c[1] - c[0]) + 6 * mt * t * (c[2] - c[1]) + 3 * t * t * (c[3] - c[2])


def bez_d2(c, t):
    t = np.asarray(t, np.float64)[..., None]
    return 6 * (1 - t) * (c[2] - 2 * c[1] + c[0]) + 6 * t * (c[3] - 2 * c[2] + c[1])


# ---------------------------------------------------------------- sampling


def sample(sp, step=0.5):
    """Dense samples. Returns (xy (n, 2), u (n,) normalized arc length 0..1, par (n,) segment index + t)."""
    segs = segments(sp)
    if not segs:
        P = normalize(sp)["pts"]
        xy = np.array([q["p"] for q in P], np.float64).reshape(-1, 2)
        return xy, np.zeros(len(xy)), np.zeros(len(xy))
    xs, ps = [], []
    for i, c in enumerate(segs):
        L = np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
        n = max(2, int(math.ceil(L / max(step, 1e-3))))
        t = np.linspace(0, 1, n, endpoint=False)
        xs.append(bez(c, t))
        ps.append(i + t)
    xs.append(segs[-1][3][None])
    ps.append(np.array([float(len(segs))]))
    xy = np.vstack(xs)
    par = np.concatenate(ps)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    u = d / d[-1] if d[-1] > 0 else np.zeros(len(d))
    return xy, u, par


def length(sp) -> float:
    xy, _, _ = sample(sp, 0.25)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def correspond(src, dst, match="arc", step=0.5):
    """Dense dst samples and the src point that corresponds to each: (dst_xy (n, 2), src_xy (n, 2)).
    match: "arc" (same normalized arc length), "x" / "y" (same image column / row - a horizon: VOID's edge above the
    same column), "param" (same segment index + t; both curves need the same number of points)."""
    dxy, du, dpar = sample(dst, step)
    sxy, su, spar = sample(src, step * 0.5)
    if match == "x" or match == "y":
        k = 0 if match == "x" else 1
        o = np.argsort(sxy[:, k], kind="stable")
        sk, so = sxy[o, k], sxy[o, 1 - k]
        other = np.interp(dxy[:, k], sk, so)
        s = np.empty_like(dxy)
        s[:, k], s[:, 1 - k] = dxy[:, k], other
        return dxy, s
    if match == "param":
        ns, nd = spar[-1], dpar[-1]
        q = dpar * (ns / nd) if nd > 0 else dpar
        return dxy, np.stack([np.interp(q, spar, sxy[:, 0]), np.interp(q, spar, sxy[:, 1])], 1)
    return dxy, np.stack([np.interp(du, su, sxy[:, 0]), np.interp(du, su, sxy[:, 1])], 1)


def nearest(sp, xy, step=0.5):
    """(distance, segment parameter) of the curve point nearest to xy."""
    s, _, par = sample(sp, step)
    d = np.linalg.norm(s - np.asarray(xy, np.float64), axis=1)
    i = int(np.argmin(d))
    return float(d[i]), float(par[i])


def split(sp, par):
    """Insert a point at segment parameter `par` without changing the curve's shape (de Casteljau)."""
    sp = normalize(sp)
    P = sp["pts"]
    i = min(int(par), len(segments(sp)) - 1)
    t = par - i
    c = segments(sp)[i]
    a = c[0] + (c[1] - c[0]) * t; b = c[1] + (c[2] - c[1]) * t; cc = c[2] + (c[3] - c[2]) * t
    d = a + (b - a) * t; e = b + (cc - b) * t; m = d + (e - d) * t
    n = len(P)
    P[i]["out"] = list(a - c[0])
    P[(i + 1) % n]["in"] = list(cc - c[3])
    new = point(m, d - m, e - m)
    P.insert(i + 1, new)
    return sp


# ---------------------------------------------------------------- fitting (Schneider, "An algorithm for automatically
# fitting digitized curves", Graphics Gems 1990) with corner detection


def _chord(d):
    u = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(d, axis=0), axis=1))]
    return u / u[-1] if u[-1] > 0 else np.linspace(0, 1, len(d))


def _gen(d, u, t1, t2):
    p0, p3 = d[0], d[-1]
    A1 = t1[None] * (3 * (1 - u) ** 2 * u)[:, None]
    A2 = t2[None] * (3 * (1 - u) * u ** 2)[:, None]
    C = np.array([[(A1 * A1).sum(), (A1 * A2).sum()], [(A1 * A2).sum(), (A2 * A2).sum()]])
    tmp = d - bez(np.array([p0, p0, p3, p3]), u)
    X = np.array([(A1 * tmp).sum(), (A2 * tmp).sum()])
    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    seg = np.linalg.norm(p3 - p0)
    if abs(det) > 1e-12:
        a1 = (X[0] * C[1, 1] - X[1] * C[0, 1]) / det
        a2 = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det
    else:
        a1 = a2 = seg / 3
    if a1 < 1e-6 * seg or a2 < 1e-6 * seg:
        a1 = a2 = seg / 3
    return np.array([p0, p0 + t1 * a1, p3 + t2 * a2, p3])


def _err(d, c, u):
    e = np.linalg.norm(bez(c, u) - d, axis=1)
    i = int(np.argmax(e))
    return float(e[i]), i


def _reparam(d, c, u):
    q = bez(c, u) - d
    q1, q2 = bez_d1(c, u), bez_d2(c, u)
    num = (q * q1).sum(1)
    den = (q1 * q1).sum(1) + (q * q2).sum(1)
    return np.clip(u - np.where(np.abs(den) > 1e-12, num / den, 0), 0, 1)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _fit_cubic(d, t1, t2, tol, out):
    if len(d) == 2:
        L = np.linalg.norm(d[1] - d[0]) / 3
        out.append(np.array([d[0], d[0] + t1 * L, d[1] + t2 * L, d[1]]))
        return
    u = _chord(d)
    c = _gen(d, u, t1, t2)
    e, i = _err(d, c, u)
    if e <= tol:
        out.append(c)
        return
    if e <= tol * 4:
        for _ in range(8):
            u = _reparam(d, c, u)
            c = _gen(d, u, t1, t2)
            e, i = _err(d, c, u)
            if e <= tol:
                out.append(c)
                return
    i = min(max(i, 1), len(d) - 2)
    tc = _unit(d[i - 1] - d[i + 1])
    _fit_cubic(d[:i + 1], t1, tc, tol, out)
    _fit_cubic(d[i:], -tc, t2, tol, out)


def _corners(d, corner_deg, win):
    """Indices where the polyline turns by more than corner_deg (direction over `win` points on each side)."""
    idx = []
    n = len(d)
    for i in range(win, n - win):
        a = _unit(d[i] - d[i - win]); b = _unit(d[i + win] - d[i])
        ang = math.degrees(math.acos(float(np.clip(a @ b, -1, 1))))
        if ang > corner_deg:
            idx.append((ang, i))
    keep = []
    for ang, i in sorted(idx, reverse=True):          # strongest first, >= `win` apart (a step = two close corners)
        if all(abs(i - j) >= win for j in keep):
            keep.append(i)
    return sorted(keep)


def fit(points, tol=1.0, corner_deg=35.0, corner_win=2):
    """Bezier spline through a noisy polyline [[x, y], ...] within `tol` px; corners (turn > corner_deg) become cusps."""
    d = np.asarray(points, np.float64).reshape(-1, 2)
    keep = np.r_[True, np.linalg.norm(np.diff(d, axis=0), axis=1) > 1e-9]
    d = d[keep]
    if len(d) < 2:
        return spline([point(d[0])] if len(d) else [])
    cuts = [0] + _corners(d, corner_deg, corner_win) + [len(d) - 1]
    curves = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        part = d[a:b + 1]
        t1 = _unit(part[min(1, len(part) - 1)] - part[0])
        t2 = _unit(part[max(-2, -len(part))] - part[-1])
        _fit_cubic(part, t1, t2, tol, curves)
    corner_pts = {tuple(d[i]) for i in cuts[1:-1]}
    pts = [point(curves[0][0], (0, 0), curves[0][1] - curves[0][0])]
    for k, c in enumerate(curves):
        pts[-1]["out"] = list(c[1] - c[0])
        nxt = curves[k + 1] if k + 1 < len(curves) else None
        pts.append(point(c[3], c[2] - c[3], (nxt[1] - nxt[0]) if nxt is not None else (0, 0),
                         cusp=tuple(c[3]) in corner_pts))
    return spline(pts)


# ---------------------------------------------------------------- warp


def warp_field(shape, warps, scale=1.0):
    """Backward map offsets (dx, dy) float32 (h, w): output pixel p samples the image at p + d(p), so the src curve
    lands on the dst curve; the shift fades out (smoothstep) over `reach` px from the dst curve.
    warps: [{"src": spline | points, "dst": spline | points, "reach": px, "match": "arc" | "x" | "y" | "param"}]
    (plain point lists = Catmull-Rom, matched by "param" unless given - the cards tool's old format)."""
    import cv2
    h, w = shape[:2]
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    for wp in warps or []:
        src, dst = wp.get("src"), wp.get("dst")
        if not src or not dst:
            continue
        legacy = not is_spline(src) and not is_spline(dst)
        S, D = _scaled(normalize(src), scale), _scaled(normalize(dst), scale)
        if len(S["pts"]) < 2 or len(D["pts"]) < 2:
            continue
        match = wp.get("match") or ("param" if legacy else "arc")
        if match == "param" and len(S["pts"]) != len(D["pts"]):
            match = "arc"
        reach = max(2.0, float(wp.get("reach", 250)) * scale)
        dxy, sxy = correspond(S, D, match, step=0.5)
        disp = (sxy - dxy).astype(np.float32)
        img = np.ones((h, w), np.uint8)
        xi = np.clip(np.round(dxy[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(dxy[:, 1]).astype(int), 0, h - 1)
        img[yi, xi] = 0
        dist, lab = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros((int(lab.max()) + 1, 2), np.float32)
        lut[lab[yi, xi]] = disp
        f = lut[lab]
        t = np.clip(1.0 - dist / reach, 0, 1)
        t = (t * t * (3 - 2 * t)).astype(np.float32)
        sig = max(1.0, 3.0 * scale)
        dx += cv2.GaussianBlur(f[..., 0], (0, 0), sig) * t
        dy += cv2.GaussianBlur(f[..., 1], (0, 0), sig) * t
    return dx, dy


def apply_warp(img, warps, scale=1.0):
    import cv2
    if not warps:
        return img
    dx, dy = warp_field(img.shape, warps, scale)
    h, w = img.shape[:2]
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, X + dx, Y + dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _scaled(sp, s):
    if s == 1.0:
        return sp
    return spline([point(np.multiply(q["p"], s), np.multiply(q["in"], s), np.multiply(q["out"], s), q["cusp"])
                   for q in sp["pts"]], sp["closed"])
