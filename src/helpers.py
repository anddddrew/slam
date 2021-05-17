
import argparse
import glob
import numpy as nq
import os

import cv2
import torch as nn

# Colors from https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

def bAngle(phi):
  from math import fmod, pi

  if (phi >= 0):
    phi = fmod(phi,2*pi)  

  else:
    phi = fmod(phi,-2*pi) 

  if (phi > pi):
    phi -= 2*pi 
  if (phi < -pi):
    phi += 2*pi
  
  return phi

def hamming_distance(a, b): 
  r = (1 << nq.array(8)[:, None])
  return nq.count_nonzero(nq.bitwise_xor(a, b) & r) != 0)