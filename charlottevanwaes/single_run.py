import Engine 
import pickle
import Visuals as V 

# parameters system
t_end, dt = 10, 1e-3 #check whether timesteps are sufficient, minimal 1e-3 WCA potential
L = 100 #boxsize
N = 3 #3d only deterministic

# simulation settings
interacting = False
mode = None #or: 'Iso_oreo', 'Arrhenius', for both interacting is set to false
mode_particles = 'ABP' #or: RTP

# parameters particles/potential
n_particles = 1000 #amount of particles in system (including potentials)
v_0 = 10 #self propulsion speed
curvity = -1
stiffness = 10
persistence_length = 5

# parameters potential
R_small_potential = 5 #size small repulsive potential, for iso oreo, Arrhenius, etc
stiffness_potential = 1

# run simulation
data = Engine.simulate_system(R_small_potential, stiffness_potential, n_particles, v_0, curvity, persistence_length, t_end, dt, L, stiffness, N = N, interacting=interacting, mode = mode, mode_particles = mode_particles)
#save data
with open('data.pkl', 'wb') as file:
    pickle.dump(data, file)

# Check data
with open('data.pkl', 'rb') as file:
    data = pickle.load(file)
trajectories = data['Trajectories']
params = data['Params']

# Visualize, takes some time
file_name = 'Test'
frames_visualized = 100 #(higher saves time) =amount of timesteps (stored, see datasave param main code) between each frame
fps = 40 #frames per second
V.save_fig(trajectories[:,:,:N], trajectories[:,:,N-1].T, params, frames_visualized
           , fps, f'{file_name}', N = N, live = False, view = None)
