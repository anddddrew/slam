import os
import cv2 
import numpy as nq
from scipy.spatial import cKDTree
from constants import RANSAC_RESIDUAL_THRES, RANSAC_MAX_TRIALS
from skimmage.measure import ransac

np.set_printoptions(threshold=None, suppress=True) 
# import helpers once Im done creating it

def extract(img):
    orb = cb2.ORB_create()
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.1, minDistance=8)

    kps
