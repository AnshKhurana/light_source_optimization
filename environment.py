from matplotlib import pyplot as plt
import numpy as np


class Room(object):

    def __init__(self, l, b, h, plane_x_intercept,
        plane_y_intercept, plane_z_intercept,
        mesh_resolution, objective_type='simple'):

        super().__init__()
        self.l = l
        self.b = b
        self.h = h
        self.plane_x_intercept = plane_x_intercept
        self.plane_y_intercept = plane_y_intercept
        self.plane_z_intercept = plane_z_intercept
        self.mesh_resolution = mesh_resolution
        self.mesh_x, self.mesh_y, self.mesh_z =  0,0,0
        self.objective_type = objective_type

    def show_plane(self):
        pass

    def return_grid(self):
        pass

    def objective_function(self, bulb_positions):
        pass
