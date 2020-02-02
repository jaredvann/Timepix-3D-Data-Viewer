import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import OpenGL.GL as GL
from PyQt5.QtCore import QTimer
from pyqtgraph.opengl import GLGridItem, GLScatterPlotItem, GLViewWidget

import constants


DEFAULT_HIT_SIZE = 4
DEFAULT_COLORMAP = "viridis"
DEFAULT_PLANE_SEPARATION = 200 #us

GRID_SPACING = 8
VIEW_DISTANCE = 512

FRAMES_PER_SECOND = 30

LIGHT_BG_COLOR = 0.9
DARK_BG_COLOR = 0.1

LIGHT_BG_GRID_COLOR = (0, 0, 0, 0.3)
DARK_BG_GRID_COLOR = (1, 1, 1, 0.3)

BLACK = (0, 0, 0, 1)
WHITE = (1, 1, 1, 1)

class Grid2(GLGridItem):
    """
    Replacement for the pyqtgraph Grid item because the color setting is broken
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "color" in kwargs:
            self.color = kwargs["color"]
        else:
            self.color = DARK_BG_GRID_COLOR


    def setColor(self, color: Tuple[float, float, float, float]):
        self.color = color
        self.update()


    def paint(self):
        self.setupGLState()

        if self.antialias:
            GL.glEnable(GL.GL_LINE_SMOOTH)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)

        GL.glBegin(GL.GL_LINES)

        x, y, _ = self.size()
        xs, ys, _ = self.spacing()
        xvals = np.arange(-x/2., x/2. + xs*0.001, xs)
        yvals = np.arange(-y/2., y/2. + ys*0.001, ys)
        GL.glColor4f(*self.color)
        for x in xvals:
            GL.glVertex3f(x, yvals[0], 0)
            GL.glVertex3f(x, yvals[-1], 0)
        for y in yvals:
            GL.glVertex3f(xvals[0], y, 0)
            GL.glVertex3f(xvals[-1], y, 0)

        GL.glEnd()


class TPX3DView(GLViewWidget):
    def __init__(self, plane_size: Tuple[int, int] = (256, 256), mark_first_hit=False):
        super().__init__()

        self.camera_rotation_timer = QTimer()

        self.sp = None
        self.hits = None
        self.colors = None

        self.light_bg = False
        self.colormap = plt.get_cmap("viridis")
        self.plane_size = plane_size
        self.mark_first_hit = mark_first_hit

        self.setBackgroundColor(DARK_BG_COLOR)

        # Set default viewing distance
        self.opts["distance"] = VIEW_DISTANCE

        self.btm_plane = Grid2()
        self.btm_plane.setSpacing(GRID_SPACING, GRID_SPACING, GRID_SPACING)
        self.btm_plane.setSize(plane_size[0]*constants.XY_CONVERSION_FACTOR, plane_size[1]*constants.XY_CONVERSION_FACTOR, 0)
        self.addItem(self.btm_plane)

        self.top_plane = Grid2()
        self.top_plane.setSpacing(GRID_SPACING, GRID_SPACING, GRID_SPACING)
        self.top_plane.setSize(plane_size[0]*constants.XY_CONVERSION_FACTOR, plane_size[1]*constants.XY_CONVERSION_FACTOR, 0)
        self.addItem(self.top_plane)

        self.btm_plane.translate(0, 0, -DEFAULT_PLANE_SEPARATION//2 * constants.Z_CONVERSION_FACTOR * 1000)
        self.top_plane.translate(0, 0, DEFAULT_PLANE_SEPARATION//2 * constants.Z_CONVERSION_FACTOR * 1000)

        self.sp = GLScatterPlotItem(pos=np.zeros((0, 3)), color=np.zeros((0, 4)), size=DEFAULT_HIT_SIZE, glOptions="translucent")
        self.addItem(self.sp)


    def draw_hits(self, hits: np.ndarray):
        self.hits = hits

        if self.hits.size > 0:
            points = np.copy(self.hits.view(np.float64).reshape(self.hits.shape + (-1,))[:, :3])
            points[:, 0] *= -1
            points[:, 2] *= -1

            self.colors = self.colormap(self.hits["tot"])

            self.colors[:, 3] = 1.0 # Sets opacity for all hits to 100%

            if self.mark_first_hit:
                # Displays first hit in cluster in a contrasting color
                self.colors[0] = BLACK if self.light_bg else WHITE

            self.sp.setData(pos=points, color=self.colors)
        else:
            self.sp.setData(pos=np.zeros((0, 3)), color=np.zeros((0, 4)))

        self.update()


    def set_colormap(self, value: str):
        self.colormap = plt.get_cmap(value)

        if self.hits is not None and self.hits.size > 0:
            self.colors = self.colormap(self.hits["tot"])

            self.colors[:, 3] = 1.0 # Sets opacity for all hits to 100%

            # Displays first hit in cluster in a contrasting color
            self.colors[0] = BLACK if self.light_bg else WHITE

            self.sp.setData(color=self.colors)

        self.update()


    def set_hit_size(self, value: int):
        self.sp.setData(size=value)
        self.update()


    def set_display_mode(self, value: str):
        self.sp.setGLOptions(value)
        self.update()


    def set_background(self, light: bool):
        self.light_bg = light

        self.setBackgroundColor(LIGHT_BG_COLOR if self.light_bg else DARK_BG_COLOR)
        self.btm_plane.setColor(LIGHT_BG_GRID_COLOR if self.light_bg else DARK_BG_GRID_COLOR)
        self.top_plane.setColor(LIGHT_BG_GRID_COLOR if self.light_bg else DARK_BG_GRID_COLOR)

        if self.colors is not None:
            if self.mark_first_hit:
                self.colors[0] = BLACK if self.light_bg else WHITE
            self.sp.setData(color=self.colors)

        self.update()


    def set_show_planes(self, value: bool):
        self.btm_plane.setVisible(value)
        self.top_plane.setVisible(value)
        self.update()


    def auto_rotate(self, value: bool):
        def rotate():
            self.opts["azimuth"] += 0.8
            self.update()

        if value and not self.camera_rotation_timer.isActive():
            self.camera_rotation_timer.timeout.connect(rotate)
            self.camera_rotation_timer.start(1000/FRAMES_PER_SECOND)
        elif not value and self.camera_rotation_timer.isActive():
            self.camera_rotation_timer.stop()


    def set_plane_separation(self, value: int):
        # Can't find a way to set exact position of existing objects,
        # so creating new ones each time.
        self.removeItem(self.btm_plane)
        self.btm_plane = Grid2()
        self.btm_plane.setSpacing(GRID_SPACING, GRID_SPACING, GRID_SPACING)
        self.btm_plane.setSize(self.plane_size[0]*constants.XY_CONVERSION_FACTOR, self.plane_size[1]*constants.XY_CONVERSION_FACTOR, 0)
        self.addItem(self.btm_plane)

        self.removeItem(self.top_plane)
        self.top_plane = Grid2()
        self.top_plane.setSpacing(GRID_SPACING, GRID_SPACING, GRID_SPACING)
        self.top_plane.setSize(self.plane_size[0]*constants.XY_CONVERSION_FACTOR, self.plane_size[1]*constants.XY_CONVERSION_FACTOR, 0)
        self.addItem(self.top_plane)

        self.btm_plane.translate(0, 0, -value//2 * constants.Z_CONVERSION_FACTOR * 1000)
        self.top_plane.translate(0, 0, value//2 * constants.Z_CONVERSION_FACTOR * 1000)
