"""Environment definition."""
from matplotlib import pyplot as plt
import jax.numpy as np
from jax import grad

class Room(object):

    def __init__(self, l, b, h,
        mesh_resolution=10, mesh_type='horizontal', plane_a=None,
        plane_b=None, plane_c=None, plane_d=None, plane_height=None,
        objective_type='simple'):
        """Setup environment variables.

        Args:
            mesh_resolution: # points per unit length
            mesh_type: 'horizontal' or 'general'
        """
        super().__init__()
        self.l = l
        self.b = b
        self.h = h
        self.plane_a = plane_a
        self.plane_b = plane_b
        self.plane_c = plane_c
        self.plane_d = plane_d
        self.mesh_resolution = mesh_resolution
        self.plane_height = plane_height
        self.mesh_type = mesh_type
        self.objective_type = objective_type
        self.mesh_x, self.mesh_y, self.mesh_z = self.generate_mesh()
        self.objective_gradient = grad(self.objective_function)

    def generate_mesh(self):
        self.mesh_x, self.mesh_y = np.meshgrid(
                np.linspace(0, self.l, self.l*self.mesh_resolution),
                np.linspace(0, self.b, self.b*self.mesh_resolution))

        if self.mesh_type == 'general':
            # TO-DO: meshgrid for generic plane given by the equation.
            assert (self.plane_a is not None) and \
                (self.plane_b is not None) and \
                (self.plane_c is not None) and \
                (self.plane_d is not None)
            self.mesh_z = (self.plane_d - self.plane_a*self.mesh_x - 
                            self.plane_b*self.mesh_y) / self.plane_c
        elif self.mesh_type == 'horizontal':
            assert self.plane_height is not None
            self.mesh_z = self.plane_height * np.ones_like(self.mesh_x)
        else:
            raise ValueError('Mesh type %s is not defined.' % self.mesh_type)

        return self.mesh_x, self.mesh_y, self.mesh_z


    def show_plane(self, bulb_positions):
        """Creates a 3D plot of the room, with placed bulbs and the
        target plane.
        """
        pass


    def return_grid(self):
        return self.mesh_x, self.mesh_y, self.mesh_z


    def intensity_grid(self, bulb_positions):
        """
        Args:
            bulb_positions: (num_bulbs * 3) np array
        """

        num_bulbs = bulb_positions.shape[0]
        I = np.zeros_like(self.mesh_x)
        
        for bi in range(num_bulbs):
            I += 1 / ((bulb_positions[bi, 0]-self.mesh_x)**2+
            (bulb_positions[bi, 1]-self.mesh_y)**2 +
            (bulb_positions[bi, 2]-self.mesh_z)**2)
        
        return I

    def objective_function(self, bulb_positions):
        
        I = self.intensity_grid(bulb_positions)

        if self.objective_type == 'simple_min':
            obj = -1 * np.min(I)
        elif self.objective_type == 'simple_std':
            obj = np.std(I)
        else:
            raise NotImplementedError('Objective function %s is not defined.'
        % self.objective_type)

        return obj

    def evaluate_gradient(self, bulb_positions):
        return self.objective_gradient(bulb_positions)
        
     
