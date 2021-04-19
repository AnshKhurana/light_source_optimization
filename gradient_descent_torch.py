import numpy as onp
from environment import Roof
import torch

from tqdm import tqdm
# signal.signal(signal.SIGINT, signal_handler)

num_bulbs=5
onp.random.seed(42)
xj = torch.FloatTensor(onp.random.randn(2 * num_bulbs)*2).view(num_bulbs,2)
xj.requires_grad = True
room = Roof(10, 20, 15, plane_height=5, objective_type='simple_std')
# import pdb; pdb.set_trace()

def to_pos(room,x):
	# print(type(x), x)
	center = torch.tensor(onp.array(room.center))
	pos = 1 / (1 + torch.exp(-x))
	pos = pos.view(-1, 2) * center * 2
	return pos

def intensity_grid_room(room, bulb_positions):
	"""
	Args:
		bulb_positions: (num_bulbs * 3) np array
	"""
	num_bulbs = bulb_positions.shape[0]
	tmp = room.mesh_x
	print(tmp)
	I = torch.zeros(tmp.shape)
	for bi in range(num_bulbs):
		I += room.intensity_constant / ((bulb_positions[bi, 0]-room.mesh_x)**2 +
					(bulb_positions[bi, 1]-room.mesh_y)**2 +
					(bulb_positions[bi, 2]-room.mesh_z)**2)
	return I

def intensity_grid(room, bulb_positions): #roof
        """
        Args:
            bulb_positions: (num_bulbs * 3) np array
        """
        num_bulbs = bulb_positions.shape[0]
        I = torch.zeros(room.mesh_x.shape)
        for bulb in bulb_positions:
            I += 100 / ((bulb[0]-torch.FloatTensor(onp.array(room.mesh_x)))**2 +
                        (bulb[1]-torch.FloatTensor(onp.array(room.mesh_y)))**2 +
                        (room.h -torch.FloatTensor(onp.array(room.mesh_z)))**2)
        return I

def objective_function(room, x):

	bulb_positions = to_pos(room,x)
	I = intensity_grid(room,bulb_positions)
	if room.objective_type == 'simple_min':
		obj = -torch.min(I)
	elif room.objective_type == 'simple_std':
		obj = torch.std(I, unbiased=True)**2
	elif room.objective_function == 'simple_penalty_min':
		obj = -1*torch.min(I) +(bulb_pos[:, 0]) 
	elif room.objective_function == 'simple_combined':
		assert room.obj_weight is not None
		obj = room.obj_weight * (torch.std(I)**2) + (room.obj_weight - 1) * torch.min(I)
	else:
		raise NotImplementedError(
			f'Objective function {room.objective_type} is not defined.')
	return obj

def rosen(X):
	"""
	This R^2 -> R^1 function should be compatible with algopy.
	http://en.wikipedia.org/wiki/Rosenbrock_function
	A generalized implementation is available
	as the scipy.optimize.rosen function
	"""
	x = X[0]
	y = X[1]
	a = 1. - x
	b = y - x*x
	return a*a + b*b*100.

def torch_SGD(xj, room, func, iters, lr,momentum=0.9, rosen_=0):
	# global xj
	optim = torch.optim.SGD([xj], lr=lr,momentum=momentum)
	errors = []
	count = 0
	for it in tqdm(range(iters)):
		count +=1
		optim.zero_grad()
		if rosen_==1:
			J = rosen(xj)
		else:
			J = objective_function(room,xj)
		J.backward()
		if torch.norm(xj.grad) < 1e-6:
			break
		
		errors.append(J.item())
		optim.step()
		
	return xj.view(-1).detach().cpu().numpy(), count

if __name__ == "__main__":
	print(xj)
	# room.show_plane(xj.clone().detach().cpu().numpy())
	print(room.objective_function(xj.clone().detach().cpu().numpy()))
	ans, _ = torch_SGD(xj,room.objective_function, iters=20000, lr=0.03,momentum=0.9)
	print(" ", room.objective_function(ans))
	room.show_plane(ans)