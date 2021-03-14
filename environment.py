"""Environment definition."""
from matplotlib import pyplot as plt
import jax.numpy as np
from scipy.optimize import minimize
from jax import grad, hessian
import numpy as onp  # original numpy


class Room:

    def __init__(self, l, b, h,
                 mesh_resolution=10, mesh_type='horizontal', plane_a=None,
                 plane_b=None, plane_c=None, plane_d=None, plane_height=None,
                 objective_type='simple_min'):
        """Setup environment variables.

        Args:
            mesh_resolution: # points per unit length
            mesh_type: 'horizontal' or 'general'
        """
        self.l = l
        self.b = b
        self.h = h
        self.center = np.array([l, b, h]) / 2
        if plane_height is None:
            plane_height = h / 3
        self.plane_height = plane_height
        self.plane_a = plane_a
        self.plane_b = plane_b
        self.plane_c = plane_c
        self.plane_d = plane_d
        self.mesh_resolution = mesh_resolution
        self.mesh_type = mesh_type
        self.objective_type = objective_type
        self.mesh_x, self.mesh_y, self.mesh_z = self.generate_mesh()
        self.J = self.objective_function
        self.gradient = grad(self.objective_function)
        self.hessian = hessian(self.objective_function)

    def generate_mesh(self):
        self.mesh_x, self.mesh_y = np.meshgrid(
            np.linspace(0, self.l, self.l*self.mesh_resolution),
            np.linspace(0, self.b, self.b*self.mesh_resolution))

        if self.mesh_type == 'general':
            # TODO: meshgrid for generic plane given by the equation.
            assert self.plane_a is not None and \
                self.plane_b is not None and \
                self.plane_c is not None and \
                self.plane_d is not None
            self.mesh_z = (self.plane_d - self.plane_a*self.mesh_x -
                           self.plane_b*self.mesh_y) / self.plane_c
        elif self.mesh_type == 'horizontal':
            assert self.plane_height is not None
            self.mesh_z = self.plane_height * np.ones_like(self.mesh_x)
        else:
            raise ValueError(f'Mesh type {self.mesh_type} is not defined.')

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
            I += 100 / ((bulb_positions[bi, 0]-self.mesh_x)**2 +
                        (bulb_positions[bi, 1]-self.mesh_y)**2 +
                        (bulb_positions[bi, 2]-self.mesh_z)**2)
        return I

    def to_pos(self, x):
        pos = 1 / (1 + np.exp(-x))
        pos = pos.reshape(-1, 3) * self.center * 2
        return pos

    def objective_function(self, x):
        assert x.ndim == 1
        bulb_positions = self.to_pos(x)
        I = self.intensity_grid(bulb_positions)
        if self.objective_type == 'simple_min':
            obj = -np.min(I)
        elif self.objective_type == 'simple_std':
            obj = np.std(I)
        elif self.objective_function == 'simple_penalty_min':
            obj = -1*np.min(I) +(bulb_pos[:, 0]) 
        else:
            raise NotImplementedError(
                f'Objective function {self.objective_type} is not defined.')
        return obj


class Roof:

    def __init__(self, l, b, h,
                 mesh_resolution=10, mesh_type='horizontal', plane_a=None,
                 plane_b=None, plane_c=None, plane_d=None, plane_height=None,
                 objective_type='simple_min'):
        """Setup environment variables.

        Args:
            mesh_resolution: # points per unit length
            mesh_type: 'horizontal' or 'general'
        """
        self.l = l
        self.b = b
        self.h = h
        self.center = np.array([l, b]) / 2
        if plane_height is None:
            plane_height = h / 3
        self.plane_height = plane_height
        self.plane_a = plane_a
        self.plane_b = plane_b
        self.plane_c = plane_c
        self.plane_d = plane_d
        self.mesh_resolution = mesh_resolution
        self.mesh_type = mesh_type
        self.objective_type = objective_type
        self.mesh_x, self.mesh_y, self.mesh_z = self.generate_mesh()
        self.J = self.objective_function
        self.gradient = grad(self.objective_function)
        self.hessian = hessian(self.objective_function)


    def generate_mesh(self):
        self.mesh_x, self.mesh_y = np.meshgrid(
            np.linspace(0, self.l, self.l*self.mesh_resolution),
            np.linspace(0, self.b, self.b*self.mesh_resolution))

        if self.mesh_type == 'general':
            # TODO: meshgrid for generic plane given by the equation.
            assert self.plane_a is not None and \
                self.plane_b is not None and \
                self.plane_c is not None and \
                self.plane_d is not None
            self.mesh_z = (self.plane_d - self.plane_a*self.mesh_x -
                           self.plane_b*self.mesh_y) / self.plane_c
        elif self.mesh_type == 'horizontal':
            assert self.plane_height is not None
            self.mesh_z = self.plane_height * np.ones_like(self.mesh_x)
        else:
            raise ValueError(f'Mesh type {self.mesh_type} is not defined.')

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
        for bulb in bulb_positions:
            I += 100 / ((bulb[0]-self.mesh_x)**2 +
                        (bulb[1]-self.mesh_y)**2 +
                        (self.h -self.mesh_z)**2)
        return I

    def to_pos(self, x):
        pos = 1 / (1 + np.exp(-x))
        pos = pos.reshape(-1, 2) * self.center * 2
        return pos

    def objective_function(self, x):
        # assert x.ndim == 1
        bulb_positions = self.to_pos(x)
        I = self.intensity_grid(bulb_positions)
        if self.objective_type == 'simple_min':
            obj = -np.min(I)
        elif self.objective_type == 'simple_std':
            obj = np.std(I)**2
        else:
            raise NotImplementedError(
                f'Objective function {self.objective_type} is not defined.')
        return obj

room = Room(10, 15, 20, plane_height=5, objective_type='simple_min')


def room_scipy(bulb_position):

    """The Rosenbrock function"""
    bulb_position = np.reshape(bulb_position, (-1, 3))
    return room.objective_function(bulb_position)


def test_scipy():
    bulb_pos = onp.array([[6, 8, 10], [2, 1, 3], [10, 10, 15]])
    print("Init bulb pos: \n", bulb_pos)
    
    res = minimize(room_scipy, bulb_pos.ravel(), method='nelder-mead',
               options={'xatol': 1e-10, 'disp': True})

    print(np.reshape(res.x, (-1, 3)))


if __name__ == '__main__':
    room = Room(10, 15, 20, plane_height=5, objective_type='simple_min')
    bulb_pos = onp.random.rand(5 * 3) * 10
    print("Init bulb pos: \n", bulb_pos)
    grid_x, grid_y, grid_z = room.return_grid()
    print("Grid Shape :")
    print(grid_x.shape, grid_y.shape, grid_z.shape)
    print("Obj at initial bulb position: ", room.objective_function(bulb_pos))
    print("Obj gradient at initial position: \n",
          room.gradient(bulb_pos))
    print("Obj hessian at initial position: \n",
          room.gradient(bulb_pos))
