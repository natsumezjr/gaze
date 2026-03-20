"""
3D viewer for one frame of gaze log: camera at origin, eye points and fitted eyeball spheres + gaze direction.
Run: python -m project.tools.gaze_log_3d_viewer --log project/log/gaze_20260304.log [--frame-index N | --frame-id ID]
Use --save path.png to save figure (e.g. in headless env set MPLBACKEND=Agg).
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from project.tools.log_parser import (
    parse_gaze_log,
    FrameData,
    FrameFitting,
    EyeFitting,
)

# Point type to (color, marker, label) for scatter
POINT_STYLE = {
    "pupil": ("#e6194b", "o", "pupil"),
    "iris": ("#3cb44b", "s", "iris"),
    "inner_canthus": ("#ffe119", "^", "inner_canthus"),
    "outer_canthus": ("#4363d8", "v", "outer_canthus"),
    "upper_eyelid": ("#f58231", ".", "upper_eyelid"),
    "lower_eyelid": ("#911eb4", ".", "lower_eyelid"),
}


def _sphere_mesh(center: tuple, radius: float, n_phi: int = 16, n_theta: int = 12):
    """Return (x,y,z) arrays for sphere surface (camera frame, mm)."""
    phi = np.linspace(0, np.pi, n_phi)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi, theta = np.meshgrid(phi, theta)
    cx, cy, cz = center
    x = cx + radius * np.sin(phi) * np.cos(theta)
    y = cy + radius * np.sin(phi) * np.sin(theta)
    z = cz + radius * np.cos(phi)
    return x, y, z


def draw_frame(
    ax: Axes3D,
    frame: FrameData,
    fitting: Optional[FrameFitting],
    scale_arrow: float = 25.0,
    pickables: Optional[list] = None,
) -> None:
    """Draw one frame: camera at origin, points by type, spheres and gaze arrows if fitting exists.
    If pickables is provided, append {artist, xs, ys, zs, label} for each scatter so click can show coordinates.
    """
    pickables = pickables if pickables is not None else []
    _append = lambda artist, xs, ys, zs, label: pickables.append({
        "kind": "point", "artist": artist, "xs": np.atleast_1d(np.asarray(xs)),
        "ys": np.atleast_1d(np.asarray(ys)), "zs": np.atleast_1d(np.asarray(zs)), "label": label,
    })
    _append_gaze = lambda artist, center, direction, label: pickables.append({
        "kind": "gaze", "artist": artist, "center": center, "direction": direction, "label": label,
    })

    # Camera at origin
    sc = ax.scatter([0], [0], [0], color="black", s=80, marker="^", label="Camera (origin)", zorder=10, picker=5)
    _append(sc, [0], [0], [0], "Camera (origin)")
    ax.text(0, 0, 0, " Camera", fontsize=8, color="black")

    # Scatter points per eye per type
    for eye in ("left", "right"):
        if eye not in frame.points:
            continue
        for ptype, pts in frame.points[eye].items():
            if not pts:
                continue
            style = POINT_STYLE.get(ptype, ("gray", "o", ptype))
            color, marker, label = style
            arr = np.array(pts)
            xs, ys, zs = arr[:, 0], arr[:, 1], arr[:, 2]
            size = 40 if ptype == "pupil" else 20
            sc = ax.scatter(xs, ys, zs, c=color, marker=marker, s=size, label=f"{eye} {label}", alpha=0.9, picker=5)
            _append(sc, xs, ys, zs, f"{eye} {label}")

    # Fitted spheres, eyeball centers (highlighted), and gaze arrows
    if fitting:
        for eye in ("left", "right"):
            ef = fitting.left if eye == "left" else fitting.right
            if not ef:
                continue
            cx, cy, cz = ef.center
            x, y, z = _sphere_mesh(ef.center, ef.radius)
            ax.plot_surface(
                x, y, z,
                color="cyan" if eye == "left" else "orange",
                alpha=0.15,
                rstride=2,
                cstride=2,
            )
            # Eyeball center: prominent marker and label
            center_color = "darkviolet" if eye == "left" else "darkgreen"
            sc = ax.scatter(
                [cx], [cy], [cz],
                c=center_color,
                s=180,
                marker="*",
                edgecolors="yellow",
                linewidths=1.5,
                label=f"{eye} eyeball center",
                zorder=10,
                picker=5,
            )
            _append(sc, [cx], [cy], [cz], f"{eye} eyeball center")
            ax.text(cx, cy, cz, f"  {eye} center", fontsize=9, color=center_color, fontweight="bold")
            # Gaze direction arrow from center (pickable)
            dx, dy, dz = ef.gaze_direction_unit
            if abs(dx) + abs(dy) + abs(dz) > 1e-6:
                qv = ax.quiver(
                    cx, cy, cz,
                    dx * scale_arrow, dy * scale_arrow, dz * scale_arrow,
                    color="red" if eye == "left" else "darkred",
                    arrow_length_ratio=0.15,
                    linewidth=1.5,
                    label=f"{eye} gaze",
                )
                qv.set_picker(8)
                _append_gaze(qv, (cx, cy, cz), (dx, dy, dz), f"{eye} gaze")

    ax.set_xlabel("X (mm, right)")
    ax.set_ylabel("Y (mm, down)")
    ax.set_zlabel("Z (mm, forward)")
    ax.set_title("Gaze frame (camera at origin, mm)")
    # Equal aspect so spheres look round
    try:
        all_pts = []
        for eye in ("left", "right"):
            if eye not in frame.points:
                continue
            for pts in frame.points[eye].values():
                all_pts.extend(pts)
        if all_pts:
            arr = np.array(all_pts)
            mx = max(np.abs(arr).max() * 1.2, 50)
            ax.set_xlim(-mx, mx)
            ax.set_ylim(-mx, mx)
            ax.set_zlim(0, arr[:, 2].max() * 1.2)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="3D eyeball viewer from gaze log (camera at origin)")
    parser.add_argument("--log", type=str, default="project/log/gaze_20260304.log", help="Path to gaze log file")
    parser.add_argument("--frame-index", type=int, default=None, help="Frame index (0-based) in parsed list")
    parser.add_argument("--frame-id", type=int, default=None, help="Frame ID to show (overrides --frame-index)")
    parser.add_argument("--scale-arrow", type=float, default=25.0, help="Gaze arrow length scale (mm)")
    parser.add_argument("--save", type=str, default=None, help="Save figure to path instead of showing (e.g. frame_3d.png)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = _project_root / log_path
    if not log_path.exists():
        print(f"Log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    frames, frame_id_to_fitting = parse_gaze_log(str(log_path))
    if not frames:
        print("No frames with eye coordinates found in log.", file=sys.stderr)
        sys.exit(1)

    # Select frame
    if args.frame_id is not None:
        idx = next((i for i, f in enumerate(frames) if f.frame_id == args.frame_id), None)
        if idx is None:
            print(f"Frame id {args.frame_id} not found. Available frame_ids: {[f.frame_id for f in frames[:10]]}...", file=sys.stderr)
            sys.exit(1)
    else:
        idx = args.frame_index if args.frame_index is not None else 0
        if idx < 0 or idx >= len(frames):
            print(f"Frame index {idx} out of range [0, {len(frames)-1}].", file=sys.stderr)
            sys.exit(1)

    frame = frames[idx]
    fitting = frame_id_to_fitting.get(frame.frame_id)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    pickables = []
    draw_frame(ax, frame, fitting, scale_arrow=args.scale_arrow, pickables=pickables)
    ax.legend(loc="upper left", fontsize=7)
    if fitting is None:
        fig.suptitle(f"Frame {frame.frame_id} (no fitting for this frame)", fontsize=10)
    else:
        fig.suptitle(f"Frame {frame.frame_id}", fontsize=10)

    # Text box for selected point coordinates (only when interactive)
    coord_text = fig.text(0.02, 0.02, "Click a point to show coordinates (mm)", fontsize=9, family="monospace",
                         verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    def on_pick(event):
        for p in pickables:
            if p["artist"] is not event.artist:
                continue
            if p.get("kind") == "gaze":
                cx, cy, cz = p["center"]
                dx, dy, dz = p["direction"]
                coord_text.set_text(
                    f"{p['label']}\n  from center (mm): x={cx:.3f} y={cy:.3f} z={cz:.3f}\n  unit vector: ({dx:.4f}, {dy:.4f}, {dz:.4f})"
                )
                fig.canvas.draw_idle()
                return
            # point
            ind = getattr(event, "ind", None)
            if ind is not None and len(ind) > 0:
                i = ind[0]
                xs, ys, zs = p["xs"], p["ys"], p["zs"]
                if i < len(xs) and i < len(ys) and i < len(zs):
                    x, y, z = float(xs[i]), float(ys[i]), float(zs[i])
                    coord_text.set_text(f"{p['label']}\n  x = {x:.3f}  y = {y:.3f}  z = {z:.3f}  (mm)")
                    fig.canvas.draw_idle()
            break

    if not args.save:
        fig.canvas.mpl_connect("pick_event", on_pick)

    plt.tight_layout(rect=(0, 0.06, 1, 1))  # leave room for coordinate text at bottom
    if args.save:
        plt.savefig(args.save, dpi=120)
        print(f"Saved to {args.save}")
    else:
        print("Tip: Drag with left mouse to rotate, right mouse to pan, scroll to zoom. Click a point to show coordinates.")
        plt.show()


if __name__ == "__main__":
    main()
