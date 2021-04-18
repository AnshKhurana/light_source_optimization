import numpy as onp
from environment import Roof
import torch
import signal
from gradient_descent import signal_handler
from tqdm import tqdm
signal.signal(signal.SIGINT, signal_handler)

num_bulbs=5
onp.random.seed(42)
xj = torch.FloatTensor(onp.random.randn(2 * num_bulbs)*2).view(num_bulbs,2)
xj.requires_grad = True
room = Roof(20, 10, 7, plane_height=5, objective_type='simple_std')
# import pdb; pdb.set_trace()
center = torch.tensor(onp.array(room.center))
def to_pos(x):
	# print(type(x), x)
	pos = 1 / (1 + torch.exp(-x))
	pos = pos.view(-1, 2) * center * 2
	return pos

def intensity_grid_room(bulb_positions):
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

def intensity_grid(bulb_positions): #roof
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

def objective_function(x):

	bulb_positions = to_pos(x)
	I = intensity_grid(bulb_positions)
	if room.objective_type == 'simple_min':
		obj = -np.min(I)
	elif room.objective_type == 'simple_std':
		obj = torch.std(I, unbiased=False)**2
	elif room.objective_function == 'simple_penalty_min':
		obj = -1*np.min(I) +(bulb_pos[:, 0]) 
	elif room.objective_function == 'simple_combined':
		assert room.obj_weight is not None
		obj = room.obj_weight * (np.std(I)**2) + (room.obj_weight - 1) * np.min(I)
	else:
		raise NotImplementedError(
			f'Objective function {room.objective_type} is not defined.')
	return obj

def torch_SGD(func, iters, lr):
	global xj
	optim = torch.optim.SGD([xj], lr=lr)
	errors = []
	for it in tqdm(range(iters)):
		optim.zero_grad()
		J = objective_function(xj)
		J.backward()
		errors.append(J.item())
		optim.step()
		with torch.no_grad():
			torch.clamp_(xj[:,0], 0, 20) #TODO Check: max x coordinate
			torch.clamp_(xj[:,1], 0, 10) #TODO Check: max y coordinate
		# tqdm.display(mean(errors))
	return xj.view(-1).detach().cpu().numpy()

if __name__ == "__main__":
	print(xj)
	room.show_plane(xj.clone().detach().cpu().numpy())
	print(room.objective_function(xj.clone().detach().cpu().numpy()))
	ans = torch_SGD(room.objective_function, iters=20000, lr=0.5)
	room.show_plane(ans)