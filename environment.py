"""Environment definition."""
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as np
from scipy.optimize import minimize
from scipy.optimize import rosen, rosen_der, rosen_hess
from jax import grad, hessian
import numpy as onp  # original numpy


class Room:

    def __init__(self, l, b, h,
                 mesh_resolution=10, mesh_type='horizontal', plane_a=None,
                 plane_b=None, plane_c=None, plane_d=None, plane_height=None,
                 obj_weight=None, transform=False, intensity_constant=100,
                 objective_type='simple_min'):
        """Setup environment variables.

        Args:
            mesh_resolution: # points per unit length
            mesh_type: 'horizontal' or 'general'
            transform: Apply sigmoid to positions to ensure location is alright
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
        self.transform = transform
        self.mesh_x, self.mesh_y, self.mesh_z = self.generate_mesh()
        self.J = self.objective_function
        self.intensity_constant = intensity_constant
        self.obj_weight = obj_weight
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
        if bulb_positions.ndim == 1:
            bulb_positions = self.to_pos(bulb_positions)

        fig = plt.figure(figsize=(10, 4))

        # Plot for Positions
        fig.suptitle('Visualisation  of Room.')
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_xlabel('l')
        ax.set_ylabel('b')
        ax.set_zlabel('h')
        ax.set_xlim(0, self.l)
        ax.set_ylim(0, self.b)
        ax.set_zlim(0, self.h)
        ax.set_title('bulb_positions')
        ax.scatter(bulb_positions[:, 0], bulb_positions[:, 1],
                   bulb_positions[:, 2], marker='^', c='r')
        ax.plot_surface(self.mesh_x, self.mesh_y, self.mesh_z)

        # Plot for intensities
        ax = fig.add_subplot(1, 2, 2)
        I = self.intensity_grid(bulb_positions)
        # im = ax.contourf(self.mesh_x, self.mesh_y, I, cmap='jet')
        ax.scatter(bulb_positions[:, 0]*self.mesh_resolution,
                   bulb_positions[:, 1]*self.mesh_resolution,
                   marker='x', c='r')
        im = ax.imshow(I)
        fig.colorbar(im)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()

    def show_itensity(self, bulb_positions):
        if bulb_positions.ndim == 1:
            bulb_positions = self.to_pos(bulb_positions)

        I = self.intensity_grid(bulb_positions)
        plt.imshow(I)
        # plt.contourf(self.mesh_x, self.mesh_y, I, cmap='jet')
        plt.colorbar()
        plt.show()

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
            I += self.intensity_constant / ((bulb_positions[bi, 0]-self.mesh_x)**2 +
                        (bulb_positions[bi, 1]-self.mesh_y)**2 +
                        (bulb_positions[bi, 2]-self.mesh_z)**2)
        return I

    def sigmoid_reshape(self, x):
        pos = 1 / (1 + np.exp(-x))
        pos = pos.reshape(-1, 3) * self.center * 2
        return pos

    def to_pos(self, x):
        if self.transform:
            return self.sigmoid_reshape(x)
        else:
            return self.simple_reshape(x)

    def simple_reshape(self, x):
        return x.reshape(-1, 3)

    def objective_function(self, x):

        assert x.ndim == 1
        bulb_positions = self.to_pos(x)
        I = self.intensity_grid(bulb_positions)
        if self.objective_type == 'simple_min':
            obj = -np.min(I)
        elif self.objective_type == 'simple_std':
            obj = np.std(I)**2
        elif self.objective_function == 'simple_penalty_min':
            obj = -1*np.min(I) +(bulb_pos[:, 0]) 
        elif self.objective_function == 'simple_combined':
            assert self.obj_weight is not None
            obj = self.obj_weight * (np.std(I)**2) + (self.obj_weight - 1) * np.min(I)
        else:
            raise NotImplementedError(
                f'Objective function {self.objective_type} is not defined.')
        return obj


class Roof:

    def __init__(self, l, b, h,
                 mesh_resolution=10, mesh_type='horizontal', plane_a=None,
                 plane_b=None, plane_c=None, plane_d=None, plane_height=None,
                 objective_type='simple_min', obj_weight=None):
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
        self.obj_weight = obj_weight
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
        if bulb_positions.ndim == 1:
            bulb_positions = self.to_pos(bulb_positions)
        heights = np.array([[self.h] * bulb_positions.shape[0]])
        bulb_pos = np.hstack([bulb_positions, heights.T])
        print(bulb_pos)
        fig = plt.figure(figsize=(10, 4))

        # Plot for Positions
        fig.suptitle('Visualisation  of Room.')
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_xlabel('l')
        ax.set_ylabel('b')
        ax.set_zlabel('h')
        ax.set_xlim(0, self.l)
        ax.set_ylim(0, self.b)
        ax.set_zlim(0, self.h)
        ax.set_title('bulb_positions')
        ax.scatter(bulb_pos[:, 0], bulb_pos[:, 1],
                   bulb_pos[:, 2], marker='^', c='r')
        ax.plot_surface(self.mesh_x, self.mesh_y, self.mesh_z)

        # Plot for intensities
        ax = fig.add_subplot(1, 2, 2)
        I = self.intensity_grid(bulb_pos)
        # im = ax.contourf(self.mesh_x, self.mesh_y, I, cmap='jet')
        ax.scatter(bulb_pos[:, 0]*self.mesh_resolution,
                   bulb_pos[:, 1]*self.mesh_resolution,
                   marker='x', c='r')
        im = ax.imshow(I)
        fig.colorbar(im)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()

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
        # print(type(x), x)
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
        elif self.objective_type == 'simple_combined':
            assert self.obj_weight is not None
            obj = self.obj_weight * (np.std(I)**2) + (self.obj_weight - 1) * np.min(I)
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


class Rosenbrock:
    def __init__(self, plot_min_x1=-5, plot_max_x1=5, plot_min_x2=-5,
                 plot_max_x2=5, mesh_resolution=10) -> None:
        self.objective_function = rosen
        self.J = self.objective_function
        self.gradient = rosen_der
        self.hessian = rosen_hess
        self.plot_min_x1 = plot_min_x1
        self.plot_max_x1 = plot_max_x1
        self.plot_min_x2 = plot_min_x2
        self.plot_max_x2 = plot_max_x2
        self.mesh_resolution = mesh_resolution
        self.mesh_x1, self.mesh_x2 = self.generate_mesh()

    def intensity_grid(self):       
        I = 100.0*(self.mesh_x1-self.mesh_x2**2.0)**2.0 + (1-self.mesh_x1)**2.0
        return I

    def generate_mesh(self):
        range_x = self.plot_max_x1 - self.plot_min_x1
        range_y = self.plot_max_x2 - self.plot_min_x2
        mesh_x, mesh_y = np.meshgrid(
            np.linspace(self.plot_min_x1, self.plot_max_x1, range_x*self.mesh_resolution),
            np.linspace(self.plot_min_x2, self.plot_max_x2, range_y*self.mesh_resolution)
            )
        return mesh_x, mesh_y

    def show_plane(self, current_pos):
        fig = plt.figure(figsize=(6, 5))
        Z = self.intensity_grid()
        ctr = plt.contour(self.mesh_x1, self.mesh_x2, Z, 50)
        plt.colorbar(ctr)
        plt.scatter(current_pos[0],
                    current_pos[1],
                    marker='x', c='r', zorder=2)
        plt.scatter(1,
                    1,
                    marker='o', c='b')
        plt.xlabel('x1')
        plt.ylabel('x2') 
        plt.title('Visualisation of the Rosenbrock function in %d to %d range' % (self.plot_min_x1, self.plot_max_x1))
        plt.show()
    

if __name__ == '__abc__':
    room = Room(10, 15, 20, plane_height=5, objective_type='simple_min')
    # bulb_pos = onp.array([6, 8, 10, 2, 1, 3, 10, 10, 15], dtype=float)
    bulb_pos = onp.random.rand(5 * 3) * 10.
    print("Init bulb pos: \n", bulb_pos)
    grid_x, grid_y, grid_z = room.return_grid()
    room.show_plane(bulb_pos)
    print("Grid Shape :")
    print(grid_x.shape, grid_y.shape, grid_z.shape)
    print("Obj at initial bulb position: ", room.objective_function(bulb_pos))
    print("Obj gradient at initial position: \n",
          room.gradient(bulb_pos))
    print("Obj hessian at initial position: \n",
          room.gradient(bulb_pos))

if __name__ == '__main__':
    x0 = np.array([-2, 3])
    rsb = Rosenbrock()
    rsb.show_plane(x0)
    print('Init pos: \n', x0)
    print('Init grad: \n', rsb.gradient(x0))
    print('Init obj: \n', rsb.J(x0))
    print('Init Hessian: \n', rsb.hessian(x0))