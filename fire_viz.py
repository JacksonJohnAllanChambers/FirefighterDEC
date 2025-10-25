##-----------------------------------------------------------------------------
# Fire Viz - Visualization of FirefighterDEC fire mapping and drone data (AI assisted) 
#-----------------------------------------------------------------------------
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from typing import Optional, Dict, List, Tuple
# -----------------------------------------------------------------------------
# Local helpers (previously imported) inlined to avoid external self-loading
# -----------------------------------------------------------------------------
WIDTH = 550
HEIGHT = 100
UNMAPPED = '\0'

DIR_TO_VEC = {
    0: (0, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0)
}

# Base stations (name, x, y)
BASE_STATIONS: List[Tuple[str, int, int]] = [
    ("YAR", 0, 40),
    ("LUN", 140, 25),
    ("WIN", 250, 70),
    ("HFX", 265, 30),
    ("NG", 400, 75),
    ("SYD", 540, 50),
]

class Drone:
    def __init__(self, x: int, y: int, name: str = "D"):
        self.x = x
        self.y = y
        self.name = name
    def location(self) -> Tuple[int, int]:
        return (self.x, self.y)

def parse_tile(s: str) -> Optional[Dict]:
    """
    Parse a tile string. Supports two formats:
    - Mapping.py convention (preferred):
        x[0:3], y[3:5], [5:reserved], fire_flag[6], severity[7], wind[8], dir[9],
        citizen[10], firefighter[11], turns_since_seen[12], trust[13]
      Minimum length: 14
    - Legacy fire_viz format:
        x[0:3], y[3:5], severity[5], wind[6], dir[7], citizen[8], firefighter[9],
        optional n/trust from position 10 onward
      Minimum length: 10
    Returns dict with unified keys: x, y, fire(severity), windspeed, winddir,
    citizen, firefighter, n(turns), trust, and fire_flag when available.
    """
    if not s or s == UNMAPPED:
        return None
    try:
        if len(s) >= 14:
            # Mapping.py convention
            x = int(s[0:3]); y = int(s[3:5])
            # s[5] reserved/unused
            fire_flag = int(s[6])
            severity = int(s[7])
            wind = int(s[8])
            wdir = int(s[9])
            citizen = int(s[10])
            firefighter = int(s[11])
            n = int(s[12])
            trust = int(s[13])
            return dict(x=x, y=y, fire=severity, fire_flag=fire_flag,
                        windspeed=wind, winddir=wdir,
                        citizen=citizen, firefighter=firefighter,
                        n=n, trust=trust)
        elif len(s) >= 10:
            # Legacy convention
            x = int(s[0:3]); y = int(s[3:5])
            severity = int(s[5])
            wind = int(s[6]); wdir = int(s[7])
            citizen = int(s[8]); firefighter = int(s[9])
            n = -1; trust = -1
            if len(s) > 10:
                tail = s[10:]
                if len(tail) == 2:
                    n, trust = int(tail[0]), int(tail[1])
                elif len(tail) == 3:
                    n, trust = int(tail[0]), int(tail[1:])
                elif len(tail) >= 4:
                    n, trust = int(tail[0:2]), int(tail[2:4])
            fire_flag = 1 if severity > 0 else 0
            return dict(x=x, y=y, fire=severity, fire_flag=fire_flag,
                        windspeed=wind, winddir=wdir,
                        citizen=citizen, firefighter=firefighter,
                        n=n, trust=trust)
        else:
            return None
    except Exception:
        return None

def encode_tile(x:int,y:int,fire:int,wind:int,wd:int,cit:int,ff:int,n:int=-1,trust:int=-1) -> str:
    """
    Encode a tile using Mapping.py convention.
    fire parameter is interpreted as severity (0-9). fire_flag is derived as 1 if fire>0 else 0.
    Positions: x[0:3], y[3:5], reserved[5]='0', fire_flag[6], severity[7], wind[8], dir[9],
               citizen[10], firefighter[11], turns[12], trust[13]
    When n/trust are negative, default them to 0 to keep fixed length.
    """
    severity = int(fire) % 10
    fire_flag = 1 if severity > 0 else 0
    wind = int(wind) % 10
    wd = int(wd) % 10  # allow up to 9; mapping.py mentions direction at index 9
    cit = int(cit) % 10
    ff = int(ff) % 10
    n = 0 if n < 0 else int(n) % 10
    trust = 0 if trust < 0 else int(trust) % 10
    return f"{x:03d}{y:02d}0{fire_flag}{severity}{wind}{wd}{cit}{ff}{n}{trust}"

def tiles_to_arrays(matrix: List[List[str]]):
    H = len(matrix); W = len(matrix[0])
    fire = np.full((H, W), np.nan)
    windspeed = np.full((H, W), np.nan)
    winddir = np.full((H, W), np.nan)
    citizen = np.zeros((H, W), dtype=bool)
    firefighter = np.zeros((H, W), dtype=bool)
    trust = np.full((H, W), np.nan)
    for y in range(H):
        for x in range(W):
            t = parse_tile(matrix[y][x])
            if t is None: continue
            fire[y, x] = t["fire"]
            windspeed[y, x] = t["windspeed"]
            winddir[y, x] = t["winddir"]
            citizen[y, x] = bool(t["citizen"])
            firefighter[y, x] = bool(t["firefighter"])
            trust[y, x] = t["trust"] if t["trust"] >= 0 else np.nan
    mapped_mask = ~np.isnan(fire)
    return {"fire": fire, "windspeed": windspeed, "winddir": winddir,
            "citizen": citizen, "firefighter": firefighter,
            "trust": trust, "mapped": mapped_mask}

def generate_dummy_maps(width=WIDTH, height=HEIGHT, seed: Optional[int]=42):
    rng = np.random.default_rng(seed)
    def empty_mat():
        return [['\\0' for _ in range(width)] for _ in range(height)]
    hist = empty_mat(); pred = empty_mat()
    clusters = [(100, 20, 6, 15), (300, 60, 7, 10), (480, 40, 5, 12)]
    for (cx, cy, base, radius) in clusters:
        for y in range(max(0, cy-radius), min(height, cy+radius+1)):
            for x in range(max(0, cx-radius), min(width, cx+radius+1)):
                dist = math.hypot(x-cx, y-cy)
                if dist <= radius:
                    sev = max(0, min(9, int(base - (dist / radius) * base + rng.integers(-1, 2))))
                    wind = int(rng.integers(2, 7)); wdir = int(rng.integers(0, 4))
                    cit = int(1 if rng.random() < 0.005 else 0); ff = int(1 if rng.random() < 0.003 else 0)
                    hist[y][x] = encode_tile(x, y, sev, wind, wdir, cit, ff, n=int(rng.integers(0, 20)), trust=int(rng.integers(0, 20)))
    for _ in range(2000):
        x = int(rng.integers(0, width)); y = int(rng.integers(0, height))
        if hist[y][x] == '\\0':
            wind = int(rng.integers(0, 4)); wdir = int(rng.integers(0, 4))
            cit = int(1 if rng.random() < 0.002 else 0); ff = int(1 if rng.random() < 0.002 else 0)
            hist[y][x] = encode_tile(x, y, 0, wind, wdir, cit, ff, n=int(rng.integers(0, 20)), trust=int(rng.integers(0, 20)))
    _DIR_TO_VEC = {0:(0,-1),1:(1,0),2:(0,1),3:(-1,0)}
    for y in range(height):
        for x in range(width):
            if hist[y][x] != '\\0':
                t = parse_tile(hist[y][x])
                sev = t["fire"]; wind = t["windspeed"]; wdir = t["winddir"]; cit = t["citizen"]; ff = t["firefighter"]
                px, py = x, y
                if sev >= 4:
                    dx, dy = _DIR_TO_VEC[wdir]
                    px = max(0, min(width-1, x + dx)); py = max(0, min(height-1, y + dy))
                    new_sev = max(0, min(9, sev // 2))
                    if pred[py][px] == '\\0':
                        pred[py][px] = encode_tile(px, py, new_sev, wind, wdir, cit, ff, n=0, trust=5)
                    else:
                        pt = parse_tile(pred[py][px])
                        merged_sev = max(pt["fire"], new_sev)
                        pred[py][px] = encode_tile(px, py, merged_sev, wind, wdir, max(cit, pt["citizen"]), max(ff, pt["firefighter"]), n=0, trust=5)
                if pred[y][x] == '\\0':
                    pred[y][x] = encode_tile(x, y, sev, wind, wdir, cit, ff, n=0, trust=8)
                else:
                    pt = parse_tile(pred[y][x])
                    merged_sev = max(pt["fire"], sev)
                    pred[y][x] = encode_tile(x, y, merged_sev, wind, wdir, max(cit, pt["citizen"]), max(ff, pt["firefighter"]), n=0, trust=8)
    return hist, pred

def generate_dummy_drones(num=4, seed: Optional[int]=7) -> List[Drone]:
    rng = np.random.default_rng(seed)
    drones = []
    for i in range(num):
        x = int(rng.integers(0, WIDTH)); y = int(rng.integers(0, HEIGHT))
        drones.append(Drone(x, y, name=chr(ord('A') + i)))
    return drones
# -----------------------------------------------------------------------------
# Simple per-round overlay plotter (static figure) for live GUI updating
# -----------------------------------------------------------------------------
def draw_scan_window(ax, x:int, y:int, color_alpha:float=0.2):
    rect = Rectangle((x-1, y-1), 3, 3, linewidth=1, edgecolor=None, facecolor='grey', alpha=color_alpha)
    ax.add_patch(rect)

def plot_fire_maps(hist_map: List[List[str]],
                   pred_map: List[List[str]],
                   drones: List[Drone],
                   show_scan_windows: bool = True,
                   pred_alpha: float = 0.35,
                   wind_stride: int = 5,
                   figsize: Tuple[int,int] = (18, 5),
                   save_path: Optional[str] = None,
                   title: str = "Firefighting Drone Swarm - Historical (base) + Predicted (overlay)"):
    arrays_hist = tiles_to_arrays(hist_map)
    arrays_pred = tiles_to_arrays(pred_map)
    H, W = arrays_hist["fire"].shape

    fig, ax = plt.subplots(figsize=figsize)

    # Base = historical fire severity, overlay = predicted
    base_fire = np.ma.masked_invalid(arrays_hist["fire"])  # nan -> transparent
    im1 = ax.imshow(base_fire, origin='upper', interpolation='nearest')
    pred_fire = np.ma.masked_invalid(arrays_pred["fire"])  # nan -> transparent
    im2 = ax.imshow(pred_fire, origin='upper', interpolation='nearest', alpha=pred_alpha)

    # Wind arrows (sampled by stride)
    for y in range(0, H, wind_stride):
        for x in range(0, W, wind_stride):
            if arrays_hist["mapped"][y, x] or arrays_pred["mapped"][y, x]:
                wd = arrays_pred["winddir"][y, x]; ws = arrays_pred["windspeed"][y, x]
                if np.isnan(wd) or np.isnan(ws):
                    wd = arrays_hist["winddir"][y, x]; ws = arrays_hist["windspeed"][y, x]
                if not (np.isnan(wd) or np.isnan(ws)):
                    dx, dy = DIR_TO_VEC[int(wd)]
                    scale = 0.2 + 0.06 * float(ws)
                    ax.arrow(x, y, dx*scale, dy*scale, head_width=0.6, head_length=0.6,
                             length_includes_head=True, zorder=3)

    # Citizens and firefighters
    citizen_mask = (arrays_hist["citizen"] | arrays_pred["citizen"])
    ff_mask = (arrays_hist["firefighter"] | arrays_pred["firefighter"])
    cy, cx = np.where(citizen_mask)
    fy, fx = np.where(ff_mask)
    ax.scatter(cx, cy, s=25, marker='^', label='Citizen')
    ax.scatter(fx, fy, s=25, marker='s', label='Firefighter')

    # Drones
    for d in drones:
        dx, dy = d.location()
        ax.scatter([dx], [dy], s=80, marker='X', label=f"Drone {d.name}")
        ax.text(dx+1, dy-1, d.name, fontsize=8, ha='left', va='center')
        if show_scan_windows:
            draw_scan_window(ax, dx, dy, color_alpha=0.15)

    ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5); ax.set_aspect('equal')
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_title(title)

    handles = [Line2D([0],[0], marker="^", linestyle="None", markersize=6, label="Citizen"),
               Line2D([0],[0], marker="s", linestyle="None", markersize=6, label="Firefighter"),
               Line2D([0],[0], marker="X", linestyle="None", markersize=8, label="Drone")]
    ax.legend(handles=handles, loc="upper right")

    cbar = fig.colorbar(im1, ax=ax, fraction=0.025, pad=0.02); cbar.set_label("Fire severity (0–9)")
    if save_path:
        plt.tight_layout(); plt.savefig(save_path, dpi=200)
    return fig, ax

# -----------------------------------------------------------------------------
# In-place updating visualizer for long-running loops
# -----------------------------------------------------------------------------
class FireViz:
    def __init__(self,
                 figsize: Tuple[int, int] = (18, 5),
                 pred_alpha: float = 0.35,
                 wind_stride: int = 5):
        self.pred_alpha = pred_alpha
        self.wind_stride = wind_stride

        # Base figure/axes
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect('equal')

        # Initialize empty layers
        base = np.full((HEIGHT, WIDTH), np.nan)
        self.im_hist = self.ax.imshow(np.ma.masked_invalid(base), origin='upper', interpolation='nearest')
        self.im_pred = self.ax.imshow(np.ma.masked_invalid(base), origin='upper', interpolation='nearest', alpha=self.pred_alpha)

        # Overlays
        self.cit_scatter = self.ax.scatter([], [], s=25, marker='^', label='Citizen')
        self.ff_scatter = self.ax.scatter([], [], s=25, marker='s', label='Firefighter')
        self.drone_scatter = self.ax.scatter([], [], s=80, marker='X', label='Drone')
        self.drone_labels: Dict[str, any] = {}
        self.scan_patches: List[Rectangle] = []
        self.wind_artists: List[any] = []

        self.ax.set_xlim(-0.5, WIDTH-0.5)
        self.ax.set_ylim(HEIGHT-0.5, -0.5)
        self.ax.set_xlabel("X"); self.ax.set_ylabel("Y")

        handles = [Line2D([0],[0], marker="^", linestyle="None", markersize=6, label="Citizen"),
                   Line2D([0],[0], marker="s", linestyle="None", markersize=6, label="Firefighter"),
                   Line2D([0],[0], marker="X", linestyle="None", markersize=8, label="Drone")]
        self.ax.legend(handles=handles, loc="upper right")
        self.cbar = self.fig.colorbar(self.im_hist, ax=self.ax, fraction=0.025, pad=0.02)
        self.cbar.set_label("Fire severity (0–9)")

    def _clear_artists(self):
        for p in self.scan_patches:
            try:
                p.remove()
            except Exception:
                pass
        self.scan_patches.clear()
        for a in self.wind_artists:
            try:
                a.remove()
            except Exception:
                pass
        self.wind_artists.clear()

    def draw(self,
             hist_map: List[List[str]],
             pred_map: List[List[str]],
             drones: List[Drone],
             title: Optional[str] = None,
             save_path: Optional[str] = None,
             show_scan_windows: bool = True):
        arrays_hist = tiles_to_arrays(hist_map)
        arrays_pred = tiles_to_arrays(pred_map)

        # Update raster layers
        self.im_hist.set_data(np.ma.masked_invalid(arrays_hist["fire"]))
        self.im_pred.set_data(np.ma.masked_invalid(arrays_pred["fire"]))

        # Update scatters
        cy, cx = np.where(arrays_hist["citizen"] | arrays_pred["citizen"])
        fy, fx = np.where(arrays_hist["firefighter"] | arrays_pred["firefighter"])
        self.cit_scatter.set_offsets(np.c_[cx, cy] if len(cx) else np.empty((0,2)))
        self.ff_scatter.set_offsets(np.c_[fx, fy] if len(fx) else np.empty((0,2)))

        # Drones and labels
        dxs, dys = [], []
        names_present = set()
        for d in drones:
            x, y = d.location()
            dxs.append(x); dys.append(y)
            names_present.add(d.name)
            if d.name not in self.drone_labels:
                self.drone_labels[d.name] = self.ax.text(x+1, y-1, d.name, fontsize=8, ha='left', va='center')
            else:
                self.drone_labels[d.name].set_position((x+1, y-1))
        # Remove labels for drones that disappeared
        for name in list(self.drone_labels.keys()):
            if name not in names_present:
                try:
                    self.drone_labels[name].remove()
                except Exception:
                    pass
                del self.drone_labels[name]
        self.drone_scatter.set_offsets(np.c_[dxs, dys] if len(dxs) else np.empty((0,2)))

        # Clear and redraw wind and scan windows
        self._clear_artists()
        if show_scan_windows:
            for d in drones:
                x, y = d.location()
                rect = Rectangle((x-1, y-1), 3, 3, linewidth=1, edgecolor=None, facecolor='grey', alpha=0.15)
                self.ax.add_patch(rect)
                self.scan_patches.append(rect)

        for y in range(0, HEIGHT, self.wind_stride):
            for x in range(0, WIDTH, self.wind_stride):
                if arrays_hist["mapped"][y, x] or arrays_pred["mapped"][y, x]:
                    wd = arrays_pred["winddir"][y, x]; ws = arrays_pred["windspeed"][y, x]
                    if np.isnan(wd) or np.isnan(ws):
                        wd = arrays_hist["winddir"][y, x]; ws = arrays_hist["windspeed"][y, x]
                    if not (np.isnan(wd) or np.isnan(ws)):
                        dx, dy = DIR_TO_VEC[int(wd)]
                        scale = 0.2 + 0.06 * float(ws)
                        arr = self.ax.arrow(x, y, dx*scale, dy*scale, head_width=0.6, head_length=0.6,
                                            length_includes_head=True, zorder=3)
                        self.wind_artists.append(arr)

        if title:
            self.ax.set_title(title)
        if save_path:
            self.fig.tight_layout()
            self.fig.savefig(save_path, dpi=200)
        # Nudge the GUI to render this update; caller can still call plt.pause()
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass
        return self.fig, self.ax

# --- helper to compute bounds that fully contain the rotated W×H rectangle ---
def rotated_bounds(w, h, deg):
    cx, cy = w / 2.0, h / 2.0
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    corners[:, 0] -= cx; corners[:, 1] -= cy
    R = np.array([[c, -s], [s, c]])
    rc = corners @ R.T
    rc[:, 0] += cx; rc[:, 1] += cy
    xmin, ymin = rc.min(axis=0)
    xmax, ymax = rc.max(axis=0)
    return xmin, xmax, ymin, ymax

# --- demo rounds generator (swap with your real per-round data) ---
def make_rounds(num_rounds=25, seed=135):
    rng = np.random.default_rng(seed)
    rounds = []
    hist, pred = generate_dummy_maps(seed=seed)
    drones = [Drone(x, y, name) for (name, x, y) in BASE_STATIONS]
    tracks = {d.name: [(d.x, d.y)] for d in drones}

    for r in range(num_rounds):
        new_pred = [row[:] for row in pred]

        # simple fire propagation
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if hist[y][x] != "\0":
                    t = parse_tile(hist[y][x])
                    if not t:
                        continue
                    sev, wdir, wind = t["fire"], t["winddir"], t["windspeed"]
                    if sev >= 4:
                        dx, dy = {0:(0,-1),1:(1,0),2:(0,1),3:(-1,0)}[wdir]
                        nx = max(0, min(WIDTH-1, x+dx))
                        ny = max(0, min(HEIGHT-1, y+dy))
                        t2 = parse_tile(new_pred[ny][nx]) if new_pred[ny][nx] != "\0" else None
                        sev2 = max(0, min(9, sev//2))
                        if t2 is None or sev2 > t2["fire"]:
                            new_pred[ny][nx] = encode_tile(nx, ny, sev2, wind, wdir,
                                                            t["citizen"], t["firefighter"], n=r, trust=5)

        # move drones (demo)
        for d in drones:
            dx, dy = int(np.random.randint(-3, 4)), int(np.random.randint(-2, 3))
            d.x = int(np.clip(d.x + dx, 0, WIDTH-1))
            d.y = int(np.clip(d.y + dy, 0, HEIGHT-1))
            tracks[d.name].append((d.x, d.y))

        pred = new_pred
        rounds.append((hist, pred, [Drone(d.x, d.y, d.name) for d in drones]))

        # reveal newly scanned tiles (3×3)
        for d in drones:
            for yy in range(max(0, d.y-1), min(HEIGHT, d.y+2)):
                for xx in range(max(0, d.x-1), min(WIDTH, d.x+2)):
                    if hist[yy][xx] == "\0":
                        hist[yy][xx] = encode_tile(xx, yy, 0,
                                                    int(np.random.randint(0, 4)),
                                                    int(np.random.randint(0, 4)),
                                                    0, 0, n=r, trust=2)
    return rounds, tracks

# --- main animation builder ---
def build_anim_data_only(
    rounds,
    tracks,
    rotate_deg=360.0,                  # data rotated CCW; 300° = 15° clockwise from 315°
    pred_alpha=0.35,
    wind_stride=8,
    out_path="fire_viz_anim_rot135_basemap.gif",
    bg_image_path: Optional[str] = None,
    bg_alpha: float = 0.55,            # basemap opacity
    bg_extent: Optional[tuple] = None, # (xmin, xmax, ymin, ymax) in data tile coordinates
):
    H, W = HEIGHT, WIDTH
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_axis_off()
    ax.set_facecolor('none')  # transparent so background basemap shows through

    # Fit the rotated data fully in view
    xmin, xmax, ymin, ymax = rotated_bounds(W, H, rotate_deg)
    pad = 5
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymax + pad, ymin - pad)  # y inverted for image-style view
    ax.set_aspect('equal')               # preserve aspect -> no squish/stretch

    # Rotation transform for data layers ONLY
    base_trans = Affine2D().rotate_deg_around(W/2.0, H/2.0, rotate_deg) + ax.transData

    # --- Basemap as full-figure background (NO rotation, not constrained to grid) ---
    if bg_image_path and os.path.exists(bg_image_path):
        img = plt.imread(bg_image_path)
        ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-100)
        ax_bg.imshow(img, origin="upper", alpha=bg_alpha)
        ax_bg.set_axis_off()
        # Note: This background fills the figure; it preserves image aspect and is not tied to grid extents.

    # First-frame arrays
    first_hist, first_pred, _ = rounds[0]
    arrays_hist = tiles_to_arrays(first_hist)
    arrays_pred = tiles_to_arrays(first_pred)

    # Fire layers (rotated)
    hist_img = ax.imshow(np.ma.masked_invalid(arrays_hist["fire"]),
                         origin='upper', interpolation='nearest',
                         zorder=-5, clip_on=False)
    pred_img = ax.imshow(np.ma.masked_invalid(arrays_pred["fire"]),
                         origin='upper', interpolation='nearest',
                         alpha=pred_alpha, zorder=-4, clip_on=False)
    hist_img.set_transform(base_trans)
    pred_img.set_transform(base_trans)

    # Overlays (rotated)
    wind_artists = []
    cit_scatter = ax.scatter([], [], s=25, marker='^', zorder=5, transform=base_trans, clip_on=False)
    ff_scatter  = ax.scatter([], [], s=25, marker='s', zorder=5, transform=base_trans, clip_on=False)
    drone_scatter = ax.scatter([], [], s=80, marker='X', zorder=6, transform=base_trans, clip_on=False)

    # Static base stations (blue squares)
    bs_x = [x for (name, x, y) in BASE_STATIONS]
    bs_y = [y for (name, x, y) in BASE_STATIONS]
    base_scatter = ax.scatter(bs_x, bs_y, s=60, marker='s', color='blue', zorder=6, transform=base_trans, clip_on=False)

    drone_labels: Dict[str, any] = {}
    path_lines: Dict[str, Line2D] = {}
    for name in tracks.keys():
        line, = ax.plot([], [], linewidth=1.5, zorder=4, transform=base_trans, clip_on=False)
        path_lines[name] = line

    # HUD (screen-aligned; comment out if you want pure data)
    hud = ax.text(0.01, 0.99, "Round 0", transform=fig.transFigure,
                  ha='left', va='top', fontsize=12,
                  bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    def update(i):
        for a in wind_artists: a.remove()
        wind_artists.clear()

        hist, pred, drones = rounds[i]
        arrays_hist = tiles_to_arrays(hist)
        arrays_pred = tiles_to_arrays(pred)

        hist_img.set_data(np.ma.masked_invalid(arrays_hist["fire"]))
        pred_img.set_data(np.ma.masked_invalid(arrays_pred["fire"]))

        cy, cx = np.where(arrays_hist["citizen"] | arrays_pred["citizen"])
        fy, fx = np.where(arrays_hist["firefighter"] | arrays_pred["firefighter"])
        cit_scatter.set_offsets(np.c_[cx, cy])
        ff_scatter.set_offsets(np.c_[fx, fy])

        dxs, dys = [], []
        for d in drones:
            x, y = d.location()
            dxs.append(x); dys.append(y)
            if d.name not in drone_labels:
                drone_labels[d.name] = ax.text(x+1, y-1, d.name, fontsize=8,
                                               ha='left', va='center',
                                               zorder=7, transform=base_trans, clip_on=False)
            else:
                drone_labels[d.name].set_position((x+1, y-1))
        drone_scatter.set_offsets(np.c_[dxs, dys])

        # Drone paths
        for name, line in path_lines.items():
            pts = tracks[name][:i+1]
            if pts:
                xs, ys = zip(*pts)
                line.set_data(xs, ys)

        # Wind arrows (rotated with data)
        DIR_TO_VEC = {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0)}
        arrays_mapped = arrays_hist["mapped"] | arrays_pred["mapped"]
        for y in range(0, H, wind_stride):
            for x in range(0, W, wind_stride):
                if arrays_mapped[y, x]:
                    wd = arrays_pred["winddir"][y, x]; ws = arrays_pred["windspeed"][y, x]
                    if np.isnan(wd) or np.isnan(ws):
                        wd = arrays_hist["winddir"][y, x]; ws = arrays_hist["windspeed"][y, x]
                    if not (np.isnan(wd) or np.isnan(ws)):
                        dx, dy = DIR_TO_VEC[int(wd)]
                        scale = 0.2 + 0.06 * float(ws)
                        arr = ax.arrow(x, y, dx*scale, dy*scale, head_width=0.6, head_length=0.6,
                                       length_includes_head=True, zorder=3, transform=base_trans, clip_on=False)
                        wind_artists.append(arr)

        hud.set_text(f"Round {i}")


    anim = FuncAnimation(fig, update, frames=len(rounds), interval=180, blit=False)
    writer = PillowWriter(fps=6)
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    return out_path

# --- run demo (provide your basemap here) ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fire Viz Animation")
    parser.add_argument("--rotate", type=float, default=300.0, help="Data rotation in degrees CCW (e.g., 300 is 15° clockwise from 315)")
    parser.add_argument("--rounds", type=int, default=25, help="Number of rounds/frames")
    parser.add_argument("--outfile", type=str, default="fire_viz_anim_rot135_basemap.gif", help="Output GIF path")
    parser.add_argument("--bg", type=str, default="bg_novascotia.jpg", help="Background image path (optional)")
    args = parser.parse_args()

    rounds, tracks = make_rounds(num_rounds=args.rounds, seed=135)
    # Set bg_image_path to your basemap file. Optionally tweak bg_extent to align.
    out = build_anim_data_only(
        rounds, tracks,
        rotate_deg=-30,
        pred_alpha=0.35,
        wind_stride=8,
        out_path=args.outfile,
        bg_image_path=args.bg,           # e.g., "./nova_scotia.png"
        bg_alpha=0.55,
        bg_extent=(0, WIDTH, 0, HEIGHT)
    )
    print(f"Saved to: {out}")
