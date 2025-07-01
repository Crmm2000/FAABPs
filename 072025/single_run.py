import Engine #contains engine
import pickle
import Visuals as V #as name implies

# parameters
R_small_potential = 5 #size small repulsive potential, for iso oreo
n_particles = 1000 #amount of particles in system (including potentials)
v_0 = 10 #self propulsion speed
curvity = -1
stiffness = 10
persistence_length = 5

t_end, dt = 10, 1e-3 #check whether timesteps are sufficient, minimal 1e-3 WCA potential
L = 200 #boxsize
N = 2 #3 in next version

interacting = True
mode = None #or: 'Iso_oreo', 'Arrhenius', for both interacting is set to false
mode_particles = 'ABP'


# run simulation
data = Engine.simulate_system(R_small_potential, n_particles, v_0, curvity, persistence_length, t_end, dt, L, stiffness, N = N, interacting=interacting, mode = mode, mode_particles = mode_particles)
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
frames_visualized = 50 #amount of timesteps (stored, see datasave param main code) between each frame
fps = 40 #frames per second
V.save_fig(trajectories[:,:,:N], trajectories[:,:,N].T, params, frames_visualized
           , fps, f'{file_name}', N = N, live = False, view = None)
