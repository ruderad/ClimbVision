"""Wall models: a 3D scan of a climbing wall, and videos mapped onto it (1.6).

Scan a wall once (Scaniverse, Stray Scanner: a coloured point cloud, PLY), and every video
filmed on that wall can be mapped onto the scan. Heights and distances then come from the
3D model in real metres, not from the picture with the body as a ruler (GUIDE §6.24).

1. `climbvision wall add NAME scan.ply` stores the scan in `data/walls/NAME/` in a standard
   frame: x to the right (facing the wall), y into the wall, z up, the floor (mat surface) at
   z = 0. It finds the floor and the wall's direction from the points, and numbers the
   colourful blobs (holds, volumes) as landmarks (`front.jpg`).
2. `climbvision wall mark NAME outputs/CLIP` writes a page: click 5-8 numbered landmarks on
   the scan's front view and the same spots on the clip's wall picture (`wall.jpg`); it
   downloads `wall_marks.csv`.
3. `climbvision wall fit NAME outputs/CLIP` finds the video camera from those pairs
   (perspective-n-point, focal length searched), then refines it automatically by matching
   the wall picture against renders of the scan (SIFT). It writes `wall_fit.json` and the
   3D position of every wall-picture pixel (`wall_xyz.npz`), and `wall_fit.jpg` to check.

`metrics` then uses the scan when `wall_fit.json` is there (`ScanMetric`).
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------------ files


def walls_dir() -> Path:
    env = os.environ.get("CLIMBVISION_WALLS")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent / "data" / "walls"
    return Path("data") / "walls"


_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
              "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1", "short": "<i2",
              "ushort": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Points (n, 3) float64 and colours (n, 3) uint8 RGB of a PLY point cloud (binary
    little-endian or ASCII; faces, if any, are ignored)."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    end = raw.index(b"\n", end) + 1
    header = raw[:end].decode("ascii", "replace").splitlines()
    fmt, n, props, in_vertex = "", 0, [], False
    for line in header:
        w = line.split()
        if not w:
            continue
        if w[0] == "format":
            fmt = w[1]
        elif w[0] == "element":
            in_vertex = w[1] == "vertex"
            if in_vertex:
                n = int(w[2])
        elif w[0] == "property" and in_vertex and w[1] != "list":
            props.append((w[2], _PLY_TYPES[w[1]]))
    names = [p[0] for p in props]
    if fmt == "ascii":
        rows = raw[end:].decode().split("\n")[:n]
        a = np.array([[float(v) for v in r.split()[:len(props)]] for r in rows])
        cols = {nm: a[:, i] for i, nm in enumerate(names)}
    else:
        dt = np.dtype(props)
        v = np.frombuffer(raw[end:end + n * dt.itemsize], dt)
        cols = {nm: v[nm] for nm in names}
    P = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    if all(k in cols for k in ("red", "green", "blue")):
        C = np.stack([cols["red"], cols["green"], cols["blue"]], 1).astype(np.uint8)
    else:
        C = np.full((n, 3), 180, np.uint8)
    ok = np.isfinite(P).all(1)
    return P[ok], C[ok]


# ------------------------------------------------------------------ the standard frame


def _up_axis(P: np.ndarray) -> np.ndarray:
    """The up direction: the end of an axis where a big flat slab of points lies (the
    floor). Returns a unit vector (one of +-x, +-y, +-z)."""
    best, up = -1, None
    for ax in range(3):
        v = P[:, ax]
        for sign in (1, -1):
            lo = np.percentile(sign * v, 1)
            k = int(np.sum(np.abs(sign * v - lo) < 0.05))
            if k > best:
                best, up = k, np.eye(3)[ax] * sign
    return up


def to_standard(P: np.ndarray, up: str = "auto") -> tuple[np.ndarray, dict]:
    """Rotate/translate a scan into the standard frame (x right facing the wall, y into
    the wall, z up, floor at z = 0). Returns (points, info with the transform)."""
    if up == "auto":
        u = _up_axis(P)
    else:
        sign = -1.0 if up.startswith("-") else 1.0
        u = np.eye(3)["xyz".index(up[-1])] * sign
    z = P @ u
    floor = float(np.percentile(z, 1))
    near = np.abs(z - floor) < 0.05
    floor = float(np.median(z[near]))
    # two horizontal directions
    a = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(u, a)
    H = np.stack([P @ a, P @ b], 1)
    wallpts = H[z - floor > 0.3]
    cov = np.cov((wallpts - wallpts.mean(0)).T)
    w, V = np.linalg.eigh(cov)
    along, normal = V[:, 1], V[:, 0]
    # into the wall = away from the floor in front of it, judged from the wall's base (an
    # overhang leans out over the floor: its upper part is in front of the base)
    base = H[(z - floor > 0.3) & (z - floor < 0.9)]
    fl = H[near].mean(0) - (base.mean(0) if len(base) > 50 else wallpts.mean(0))
    if fl @ normal > 0:
        normal = -normal
    # x to the right when facing the wall (looking along +y with z up: right = y x z)
    ex = a * along[0] + b * along[1]
    ey = a * normal[0] + b * normal[1]
    if np.dot(np.cross(ey, u), ex) < 0:
        ex = -ex
    R = np.stack([ex, ey, u])  # rows: new axes
    Q = P @ R.T
    Q[:, 2] -= floor
    off = np.array([np.median(Q[:, 0]), np.percentile(Q[Q[:, 2] > 0.3, 1], 50), 0.0])
    Q -= off
    return Q, {"R": R.tolist(), "offset": off.tolist(), "floor_raw": floor,
               "up": u.tolist()}


# ------------------------------------------------------------------ landmarks


def find_landmarks(P: np.ndarray, C: np.ndarray, min_pts: int = 120,
                   vox: float = 0.03) -> list[dict]:
    """Colourful blobs (holds, volumes) as numbered landmarks: points with saturated colour,
    grouped by voxels into connected blobs; the centre of each blob."""
    from scipy import ndimage

    hsv = cv2.cvtColor(C[None, :, ::-1].astype(np.uint8), cv2.COLOR_BGR2HSV)[0].astype(float)
    colourful = (hsv[:, 1] > 110) & (hsv[:, 2] > 60) & (P[:, 2] > 0.05)
    Q, K = P[colourful], hsv[colourful]
    if len(Q) < min_pts:
        return []
    ij = np.floor((Q - Q.min(0)) / vox).astype(int)
    grid = np.zeros(ij.max(0) + 1, np.int32)
    # one hue family per blob: split red / green / blue / yellow before connecting
    fam = ((K[:, 0] + 8) % 180 // 30).astype(int)
    out = []
    for f in np.unique(fam):
        sel = fam == f
        grid[:] = 0
        grid[tuple(ij[sel].T)] = 1
        lab, n = ndimage.label(grid, np.ones((3, 3, 3)))
        L = lab[tuple(ij[sel].T)]
        for k in range(1, n + 1):
            s = Q[sel][L == k]
            if len(s) < min_pts or np.ptp(s, axis=0).max() > 1.2:
                continue  # too small, or a big volume / wall panel
            out.append({"xyz": s.mean(0).round(4).tolist(), "n": int(len(s))})
    out.sort(key=lambda d: (-round(d["xyz"][2] * 2), d["xyz"][0]))  # top to bottom
    for i, d in enumerate(out, 1):
        d["id"] = i
    return out


def front_view(P: np.ndarray, C: np.ndarray, landmarks: list[dict] | None = None,
               res: float = 160.0) -> tuple[np.ndarray, dict]:
    """Orthographic view of the wall from the front (x right, z up), nearest points on
    top; numbered landmarks and a 0.5 m grid. Returns (BGR image, mapping info)."""
    x0, x1 = np.percentile(P[:, 0], [0.5, 99.5])
    z0, z1 = -0.05, float(np.percentile(P[:, 2], 99.5)) + 0.1
    W, H = int((x1 - x0) * res) + 1, int((z1 - z0) * res) + 1
    o = np.argsort(-P[:, 1])  # far (into the wall) first, near last
    xs = ((P[o, 0] - x0) * res).astype(int)
    ys = ((z1 - P[o, 2]) * res).astype(int)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    img = np.zeros((H, W, 3), np.uint8)
    img[ys[ok], xs[ok]] = C[o][ok][:, ::-1]
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))
    for zm in np.arange(0, z1, 0.5):
        y = int((z1 - zm) * res)
        cv2.line(img, (0, y), (W, y), (90, 90, 90), 1)
        cv2.putText(img, f"{zm:.1f} m", (4, y - 4), 0, 0.45, (200, 200, 200), 1)
    info = {"x0": float(x0), "z1": float(z1), "res": res, "size": [W, H]}
    for d in landmarks or []:
        u, v = int((d["xyz"][0] - x0) * res), int((z1 - d["xyz"][2]) * res)
        cv2.circle(img, (u, v), 9, (0, 0, 0), 3)
        cv2.circle(img, (u, v), 9, (0, 255, 255), 2)
        cv2.putText(img, str(d["id"]), (u + 10, v + 5), 0, 0.6, (0, 0, 0), 4)
        cv2.putText(img, str(d["id"]), (u + 10, v + 5), 0, 0.6, (0, 255, 255), 2)
    return img, info


# ------------------------------------------------------------------ the wall model


@dataclass
class WallModel:
    name: str
    P: np.ndarray  # (n, 3) standard frame, metres
    C: np.ndarray  # (n, 3) uint8 RGB
    landmarks: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def folder(self) -> Path:
        return walls_dir() / self.name

    def save(self, folder: Path | None = None) -> Path:
        folder = Path(folder) if folder else self.folder
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "wall.npz", P=self.P.astype(np.float32), C=self.C)
        (folder / "wall.json").write_text(json.dumps(
            {"name": self.name, "landmarks": self.landmarks, **self.info}, indent=2))
        img, _ = front_view(self.P, self.C, self.landmarks)
        cv2.imwrite(str(folder / "front.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return folder

    @classmethod
    def load(cls, name: str, folder: Path | None = None) -> WallModel:
        folder = Path(folder) if folder else walls_dir() / name
        z = np.load(folder / "wall.npz")
        info = json.loads((folder / "wall.json").read_text())
        lm = info.pop("landmarks", [])
        info.pop("name", None)
        return cls(name, z["P"].astype(np.float64), z["C"], lm, info)

    @classmethod
    def from_scan(cls, name: str, ply: str | Path, up: str = "auto") -> WallModel:
        P, C = load_ply(ply)
        Q, info = to_standard(P, up)
        info["source"] = str(ply)
        info["points"] = int(len(Q))
        info["height_m"] = round(float(np.percentile(Q[:, 2], 99.5)), 2)
        info["width_m"] = round(float(np.ptp(np.percentile(Q[:, 0], [0.5, 99.5]))), 2)
        return cls(name, Q, C, find_landmarks(Q, C), info)

    def landmark(self, i: int) -> np.ndarray:
        for d in self.landmarks:
            if d["id"] == i:
                return np.array(d["xyz"], float)
        raise KeyError(f"no landmark {i} on wall {self.name}")


# ------------------------------------------------------------------ rendering and fitting


def render(model: WallModel, M: np.ndarray, R: np.ndarray, t: np.ndarray, w: int, h: int,
           splat: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Render the scan through a camera: pixel = M @ (R X + t) (M: 3x3, the intrinsics,
    possibly composed with the wall-picture transform). Returns (BGR image, (h, w, 3) XYZ
    of the nearest scan point per pixel, NaN where none)."""
    Xc = model.P @ R.T + t
    p = Xc @ M.T
    z = Xc[:, 2]
    m = z > 0.1
    u = np.round(p[m, 0] / p[m, 2]).astype(int)
    v = np.round(p[m, 1] / p[m, 2]).astype(int)
    zz, cc, pp = z[m], model.C[m], model.P[m]
    ok = (u > -splat) & (u < w + splat) & (v > -splat) & (v < h + splat)
    u, v, zz, cc, pp = u[ok], v[ok], zz[ok], cc[ok], pp[ok]
    o = np.argsort(-zz)
    u, v, zz, cc, pp = u[o], v[o], zz[o], cc[o], pp[o]
    img = np.zeros((h, w, 3), np.uint8)
    X = np.full((h, w, 3), np.nan, np.float32)
    D = np.full((h, w), np.inf, np.float32)
    for dy in range(-splat + 1, splat):
        for dx in range(-splat + 1, splat):
            a, b = u + dx, v + dy
            k = (a >= 0) & (a < w) & (b >= 0) & (b < h)
            a, b, zk = a[k], b[k], zz[k]
            closer = zk < D[b, a]
            img[b[closer], a[closer]] = cc[k][closer][:, ::-1]
            X[b[closer], a[closer]] = pp[k][closer]
            D[b[closer], a[closer]] = zk[closer]
    return img, X


@dataclass
class CameraFit:
    K: np.ndarray  # 3x3, video frame pixels
    R: np.ndarray
    t: np.ndarray
    marks_rms_px: float  # reprojection error of the marked pairs
    refined: bool = False
    matches: int = 0  # automatic matches used in the refinement (inliers)
    match_err_px: float = float("nan")

    @property
    def camera_m(self) -> np.ndarray:
        return -self.R.T @ self.t

    def to_json(self) -> dict:
        return {"K": self.K.round(3).tolist(), "R": self.R.round(6).tolist(),
                "t": self.t.round(5).tolist(), "marks_rms_px": round(self.marks_rms_px, 1),
                "refined": self.refined, "matches": self.matches,
                "match_err_px": round(self.match_err_px, 1)
                if np.isfinite(self.match_err_px) else None,
                "camera_m": self.camera_m.round(3).tolist(),
                "focal_px": round(float(self.K[0, 0]), 1)}

    @classmethod
    def from_json(cls, d: dict) -> CameraFit:
        return cls(np.array(d["K"]), np.array(d["R"]), np.array(d["t"]), d["marks_rms_px"],
                   d.get("refined", False), d.get("matches", 0),
                   d.get("match_err_px") or float("nan"))


def _intrinsics(f: float, frame: tuple[int, int]) -> np.ndarray:
    return np.array([[f, 0, frame[0] / 2], [0, f, frame[1] / 2], [0, 0, 1.0]])


def fit_from_marks(P3: np.ndarray, p2: np.ndarray, frame: tuple[int, int],
                   focal_range: tuple[float, float] = (0.4, 2.0)) -> CameraFit:
    """Camera from >= 5 pairs (3D scan point, frame pixel), searching the focal length
    (as a fraction of the frame's long side) for the smallest reprojection error."""
    if len(P3) < 5:
        raise ValueError("need at least 5 marked pairs (6-8 spread over the wall is better)")
    long = max(frame)
    best = None
    for frac in np.arange(focal_range[0], focal_range[1] + 1e-9, 0.02):
        K = _intrinsics(frac * long, frame)
        ok, rv, tv = cv2.solvePnP(P3, p2, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        rv, tv = cv2.solvePnPRefineLM(P3, p2, K, None, rv, tv)
        pr, _ = cv2.projectPoints(P3, rv, tv, K, None)
        rms = float(np.sqrt(np.mean(np.sum((pr[:, 0] - p2) ** 2, 1))))
        R = cv2.Rodrigues(rv)[0]
        in_front = np.all((P3 @ R.T + tv[:, 0])[:, 2] > 0)
        if in_front and (best is None or rms < best.marks_rms_px):
            best = CameraFit(K, R, tv[:, 0], rms)
    if best is None:
        raise ValueError("no camera fits the marked pairs")
    return best


def refine(model: WallModel, fit: CameraFit, wall_img: np.ndarray, T: np.ndarray,
           frame: tuple[int, int], P3: np.ndarray, p2: np.ndarray,
           rounds: int = 3) -> CameraFit:
    """Improve a camera fit by matching the wall picture against renders of the scan from
    that camera (SIFT), then perspective-n-point on the matches. Kept only while it
    agrees with the marks (their error at most doubles, and stays < 2 % of the frame)."""
    sift = cv2.SIFT_create(6000)
    gw = cv2.cvtColor(wall_img, cv2.COLOR_BGR2GRAY)
    kw, dw = sift.detectAndCompute(gw, (wall_img.max(2) > 0).astype(np.uint8))
    if dw is None or len(kw) < 20:
        return fit
    Ti = np.linalg.inv(T)
    h, w = wall_img.shape[:2]
    cur = fit
    limit = max(2 * fit.marks_rms_px, 0.02 * max(frame))
    for _ in range(rounds):
        img, X = render(model, T @ cur.K, cur.R, cur.t, w, h)
        kr, dr = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                       (img.max(2) > 0).astype(np.uint8))
        if dr is None or len(kr) < 20:
            break
        ms = cv2.BFMatcher().knnMatch(dw, dr, k=2)
        good = [m for m, n in (p for p in ms if len(p) == 2) if m.distance < 0.8 * n.distance]
        if len(good) < 12:
            break
        pw = np.float32([kw[m.queryIdx].pt for m in good])
        pr = np.float32([kr[m.trainIdx].pt for m in good])
        x3 = X[np.clip(pr[:, 1].astype(int), 0, h - 1), np.clip(pr[:, 0].astype(int), 0, w - 1)]
        ok = np.isfinite(x3).all(1)
        if ok.sum() < 12:
            break
        q = np.c_[pw[ok], np.ones(int(ok.sum()))] @ Ti.T
        p2m = (q[:, :2] / q[:, 2:]).astype(np.float64)
        p3m = x3[ok].astype(np.float64)
        best = None
        for df in (-0.06, -0.03, 0.0, 0.03, 0.06):
            K = _intrinsics(cur.K[0, 0] * (1 + df), frame)
            rv0 = cv2.Rodrigues(cur.R)[0]
            okp, rv, tv, inl = cv2.solvePnPRansac(p3m, p2m, K, None, rv0, cur.t.reshape(3, 1),
                                                  useExtrinsicGuess=True,
                                                  reprojectionError=0.01 * max(frame),
                                                  iterationsCount=3000)
            if not okp or inl is None or len(inl) < 12:
                continue
            ii = inl[:, 0]
            rv, tv = cv2.solvePnPRefineLM(p3m[ii], p2m[ii], K, None, rv, tv)
            prj, _ = cv2.projectPoints(p3m[ii], rv, tv, K, None)
            err = float(np.median(np.linalg.norm(prj[:, 0] - p2m[ii], axis=1)))
            pk, _ = cv2.projectPoints(P3, rv, tv, K, None)
            mrms = float(np.sqrt(np.mean(np.sum((pk[:, 0] - p2) ** 2, 1))))
            if mrms > limit:
                continue
            cand = CameraFit(K, cv2.Rodrigues(rv)[0], tv[:, 0], mrms, True, len(ii), err)
            if best is None or cand.matches > best.matches:
                best = cand
        if best is None or best.matches <= cur.matches:
            break
        cur = best
    return cur


# ------------------------------------------------------------------ the scan as a ruler


def _nan_blur(X: np.ndarray, sigma: float) -> np.ndarray:
    ok = np.isfinite(X).all(2).astype(np.float32)
    Xz = np.where(ok[..., None] > 0, X, 0).astype(np.float32)
    num = cv2.GaussianBlur(Xz, (0, 0), sigma)
    den = cv2.GaussianBlur(ok, (0, 0), sigma)[..., None]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den[..., 0] < 0.2] = np.nan
    return out


@dataclass
class ScanMetric:
    """Wall coordinates (pixels of the first frame's view) -> metres on the scan: x along
    the wall (right), y up = height above the floor (mat surface), z = 0.

    `xyz` is the 3D point of the scan seen at each wall-picture pixel (`T`: wall coords ->
    wall-picture pixels). Holds and contacts use it directly; the body (COM, joints) uses
    a smoothed version (`smooth`): the body is in front of the wall, and the surface behind
    it (a volume's edge) shouldn't make it jump."""

    xyz: np.ndarray  # (h, w, 3) float32
    smooth: np.ndarray
    T: np.ndarray
    wall: str = ""
    info: dict = field(default_factory=dict)
    ground_y_m: float = 0.0
    k: float = 1.0
    a: float = 0.0
    b: float = 0.0
    s: float = 1.0
    per_m: float = 1.0  # wall px per metre, typical (for thresholds)

    @property
    def corrected(self) -> bool:
        return True

    def _lookup(self, p: np.ndarray, which: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        shp = p.shape[:-1]
        q = p.reshape(-1, 2)
        h, w = which.shape[:2]
        out = np.full((len(q), 3), np.nan)
        ok = np.isfinite(q).all(1)
        if ok.any():
            qq = np.c_[q[ok], np.ones(int(ok.sum()))] @ self.T.T
            u = np.round(qq[:, 0] / qq[:, 2]).astype(int)
            v = np.round(qq[:, 1] / qq[:, 2]).astype(int)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            vals = np.full((len(u), 3), np.nan)
            vals[inside] = which[v[inside], u[inside]]
            out[np.flatnonzero(ok)] = vals
        return out.reshape(*shp, 3)

    def xyz_at(self, p: np.ndarray, contact: bool = True) -> np.ndarray:
        """3D scan point (x, y into the wall, z up) behind wall-coordinate point(s) p."""
        return self._lookup(p, self.xyz if contact else self.smooth)

    def to_m(self, p: np.ndarray) -> np.ndarray:
        X = self._lookup(p, self.smooth)
        return np.stack([X[..., 0], X[..., 2]], -1)

    def dist_m(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.to_m(p) - self.to_m(q), axis=-1)

    def height_m(self, p: np.ndarray) -> np.ndarray:
        return self.to_m(p)[..., 1]

    def wall_angle_deg(self, p: np.ndarray, radius_px: float = 12.0) -> float:
        """The wall's angle at wall-coordinate point p, from vertical: + overhanging
        (leaning towards the climber), - slab. NaN where the scan has no surface."""
        h, w = self.xyz.shape[:2]
        q = self.T @ np.array([p[0], p[1], 1.0])
        u, v = q[0] / q[2], q[1] / q[2]
        r = int(round(radius_px))
        u0, u1 = int(max(0, u - r)), int(min(w, u + r + 1))
        v0, v1 = int(max(0, v - r)), int(min(h, v + r + 1))
        pts = self.xyz[v0:v1, u0:u1].reshape(-1, 3).astype(float)
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) < 20:
            return float("nan")
        c = np.median(pts, 0)
        pts = pts[np.linalg.norm(pts - c, axis=1) < 0.3]  # one surface (no far background)
        if len(pts) < 20:
            return float("nan")
        _, _, vt = np.linalg.svd(pts - pts.mean(0))
        n = vt[2] * (-1 if vt[2][1] > 0 else 1)  # the side facing the climber (-y)
        return float(np.degrees(-np.arcsin(np.clip(n[2], -1, 1))))

    def summary(self) -> dict:
        return {"scale_from": f"wall scan '{self.wall}'", "perspective_corrected": True,
                **self.info}


def load_scan_metric(result_dir: str | Path) -> ScanMetric | None:
    """The scan mapping of a results folder (from `climbvision wall fit`), or None."""
    result_dir = Path(result_dir)
    f, m = result_dir / "wall_fit.json", result_dir / "wall_xyz.npz"
    if not (f.exists() and m.exists()):
        return None
    d = json.loads(f.read_text())
    z = np.load(m)
    xyz = z["xyz"].astype(np.float32)
    T = np.array(d["T"], float)
    sm = _nan_blur(xyz, max(3.0, 0.015 * max(xyz.shape[:2])))
    # typical wall px per metre around the middle of the picture
    h, w = xyz.shape[:2]
    Ti = np.linalg.inv(T)
    c = np.array([w / 2, h / 2, 1.0])
    pts = []
    for du, dv in ((0, 0), (40, 0), (0, 40)):
        q = Ti @ (c + [du, dv, 0])
        pts.append(q[:2] / q[2])
    met = ScanMetric(xyz, sm, T, d.get("wall", ""), {"wall_fit": {
        k: d["fit"][k] for k in ("marks_rms_px", "matches", "match_err_px", "focal_px")}})
    a, b, cc = (met.to_m(p) for p in pts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dpx = [np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[0])]
        dm = [np.linalg.norm(b - a), np.linalg.norm(cc - a)]
        r = [x / y for x, y in zip(dpx, dm, strict=True) if np.isfinite(y) and y > 0]
    met.per_m = float(np.median(r)) if r else 1.0
    return met


def fit_clip(model: WallModel, result_dir: str | Path, marks: list[tuple[int, float, float]],
             do_refine: bool = True) -> dict:
    """Fit a clip's camera to the wall model from marked pairs (landmark id, wall-picture
    x, y), refine, and write wall_fit.json, wall_xyz.npz and wall_fit.jpg."""
    from climbvision.engine.camera import WallImage

    result_dir = Path(result_dir)
    wi = WallImage.load(result_dir)
    summ = json.loads((result_dir / "summary.json").read_text())
    frame = tuple(int(v) for v in summ["frame_size"])
    T = np.asarray(wi.T, float)
    Ti = np.linalg.inv(T)
    P3 = np.array([model.landmark(i) for i, _, _ in marks], float)
    q = np.c_[np.array([(x, y) for _, x, y in marks], float), np.ones(len(marks))] @ Ti.T
    p2 = q[:, :2] / q[:, 2:]
    fit = fit_from_marks(P3, p2, frame)
    if do_refine:
        fit = refine(model, fit, wi.image, T, frame, P3, p2)
    h, w = wi.image.shape[:2]
    img, X = render(model, T @ fit.K, fit.R, fit.t, w, h)
    np.savez_compressed(result_dir / "wall_xyz.npz", xyz=X.astype(np.float16))
    out = {"wall": model.name, "T": T.tolist(), "frame_size": list(frame),
           "fit": fit.to_json(), "marks": [list(m) for m in marks]}
    (result_dir / "wall_fit.json").write_text(json.dumps(out, indent=2))
    # a picture to check: the wall picture, the scan over it, and height bands
    over = cv2.addWeighted(wi.image, 0.5, img, 0.5, 0)
    hm = wi.image.copy()
    Z = X[..., 2]
    col = cv2.applyColorMap(np.clip(np.nan_to_num(Z, nan=0) / max(4.5, np.nanmax(Z) or 1)
                                    * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    m = np.isfinite(Z)
    hm[m] = cv2.addWeighted(wi.image, 0.45, col, 0.55, 0)[m]
    for zz in np.arange(0.5, np.nanmax(Z) if m.any() else 0, 0.5):
        band = np.abs(np.nan_to_num(Z, nan=-9) - zz) < 0.012
        hm[band] = (255, 255, 255)
        ys, xs = np.nonzero(band)
        if len(xs):
            cv2.putText(hm, f"{zz:.1f} m", (int(np.median(xs)) - 20, int(np.median(ys)) - 4), 0,
                        0.5, (255, 255, 255), 2)
    for (i, x, y) in marks:
        pk = T @ fit.K @ (fit.R @ model.landmark(i) + fit.t)
        cv2.circle(over, (int(x), int(y)), 6, (0, 255, 255), 2)
        cv2.line(over, (int(x), int(y)), (int(pk[0] / pk[2]), int(pk[1] / pk[2])), (0, 0, 255), 2)
    cv2.imwrite(str(result_dir / "wall_fit.jpg"), np.hstack([over, hm]))
    return out
