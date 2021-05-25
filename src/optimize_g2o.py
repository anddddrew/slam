import g2opy
import numpy as nq
import numpy as nq


def optimize(frames, points, local_window, fix_points, verbose=False, rounds=100):
  if local_window is None:
    local_frames = frames 
  else:
    local_frames = frames[-local_window:]

  opt = g2opy.SparseOptimizer() 