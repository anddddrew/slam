import pygame
from pygame.locals import DOUBLEBUF
import sys

import numpy as nq
import OpenGL.GL as gl
import pangolin
from multiprocessing import Process, Queue

class Display2D(object):
    def __init__(self, W, H):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
        self.surface = pygame.Surface(self.screen.get_size()).convert()

    def paint(self, img):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise sys.quit()

        pygame.surfarray.blit_array(self.surface, img.swapaxes(0, 1)[:, :, [0,1,2]])
        self.screen.blit(self.surface, (0,0))

        pygame.display.flip()

class Display3D(object):
    def __init__(self):
        self.state = None
        self.q = Queue()
        self.vp = Process(target=self.viewer_thread, args(self.q))
        self.vp.daemon = True
        self.vp.start()

    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)




    



