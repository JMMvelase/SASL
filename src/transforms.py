# src/transforms.py
import numpy as np, math

FRAME_DIM = 63  # 21*(x,y,z)
PTS = 21

def ensure_63_from_xy(flat_xy):
    """If input is 42 (x0..x20,y0..y20) -> return 63 with z=0"""
    arr = np.array(flat_xy, dtype=np.float32)
    if arr.size == 42:
        xs = arr[:21]
        ys = arr[21:]
        zs = np.zeros(21, dtype=np.float32)
        return np.concatenate([xs, ys, zs])
    if arr.size == 63:
        return arr.astype(np.float32)
    raise ValueError(f"Bad size {arr.size}")

def reorder_to_training_order(flat_xyz_interleaved):
    """
    If input is interleaved [x0,y0,z0,x1,y1,z1,...], convert to training order:
    [x0..x20, y0..y20, z0..z20].
    """
    arr = np.array(flat_xyz_interleaved, dtype=np.float32)
    if arr.size != 63:
        raise ValueError("expected 63")
    pts = arr.reshape(21,3)
    xs = pts[:,0]
    ys = pts[:,1]
    zs = pts[:,2]
    return np.concatenate([xs, ys, zs]).astype(np.float32)

def normalize_by_wrist(flat63):
    pts = flat63.reshape(21,3).astype(np.float32)
    wx, wy, wz = pts[0]
    mx, my, mz = pts[9]  # middle_mcp
    scale = math.hypot(mx-wx, my-wy)
    if scale < 1e-6:
        dists = np.linalg.norm(pts - np.array([wx,wy,wz]), axis=1)
        scale = max(1e-3, dists.max())
    pts[:,0] = (pts[:,0] - wx) / scale
    pts[:,1] = (pts[:,1] - wy) / scale
    pts[:,2] = (pts[:,2] - wz) / scale
    return pts.flatten().astype(np.float32)

def canonicalize_left_hand(flat63):
    """Flip X axis for left hand to map to right-hand canonical pose."""
    pts = flat63.reshape(21,3).copy()
    pts[:,0] = -pts[:,0]   # or 1 - x if in [0,1] normalized image coords; choose consistent scheme
    return pts.flatten().astype(np.float32)
