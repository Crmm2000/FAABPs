#import packages
import numpy as np
import math
from tqdm import tqdm
import utils as u 

# Main simulation loop
def simulate_system(radius_repulsive_potential, n_particles, v_0, curvity, persistence_length=5, t_end = 100, dt = 1e-2, boundary = 20, stiffness = 1, interacting = False, mode = None, N = 2, mode_particles = 'ABP'):
    # Initial settings 
    t = np.arange(0, t_end, dt) # time array
    n, timestep_end = len(t), len(t) #timesteps, time_step end tracks the timesteps for the arrhenius

    # setting parameters -> adjustable to heterogeneous particles: np.random.rand(n_particles, 1) 
    v0 = np.ones((n_particles, 1)) * v_0 #self propulsion speed
    b = np.ones((n_particles,1)) #particle radius #n
    kappa = np.ones((n_particles, 1)) * curvity
    stiffness = np.ones((n_particles, 1)) * stiffness
    n_potentials = 0 # sets n_potentials, don't change

    # Rotational diffusion
    Dr = v_0/persistence_length if persistence_length != 0 else None
 
    # Bookkeeping
    trajectories = np.zeros((n_particles, n, 2*N-1)) #shape: (n_particles, steps, DOF)
    params_particles = np.column_stack([v0, b, kappa, stiffness]) #shape (n_particles, 4)
    #shape (n_particles, 4), stores all params for the particles, stiffness is specific for the repulsive potential
    params_system = boundary, t_end, n_particles, persistence_length, radius_repulsive_potential, dt, n_potentials, N 
    
    # Initial conditions
    x_0 = np.random.uniform(0, boundary, size=(n_particles, N))
    rotation_start = np.random.uniform(0, 2*np.pi, size=(n_particles, N-1))
    trajectories[:,0,:N] = x_0
    trajectories[:,0,N:]= rotation_start

    #sets experiment up if indicated, overwrites previously set initial conditions
    if mode != None:
        storage_occupancy, params_particles, params_system = u.setup_experiment(mode, params_particles, params_system, trajectories)

    #main simulation loop
    for i in tqdm(range(n-1)):  # Loop through timesteps
        
        # Position and orientation update with Euler integration (vectorized)
        cos_theta, sin_theta = np.cos(trajectories[:, i, N]), np.sin(trajectories[:, i, N]) 
        if N ==2:
            e = np.stack([cos_theta, sin_theta], axis=1) #shape e (n_particles, 2)            
        #add extra degree of freedom in orientation for 3D
        elif N ==3:
            cos_phi, sin_phi = np.cos(trajectories[:, i, N+1]), np.sin(trajectories[:, i, N+1])
            e = np.stack([sin_phi * cos_theta, sin_phi * sin_theta, cos_phi], axis=1) #shape e (n_particles, 3)  
            
        #update positions and orientations
        trajectories[:, i+1, :N] = update_positions(trajectories[:, i, :N], e, params_particles, params_system, interacting=interacting, mode=mode) 
        particles_vel = (trajectories[:, i+1, :N] - trajectories[:, i, :N]) / dt #calculate velocity before the periodic boundary condition
        trajectories[:, i+1, :N] = u.pbc_vec(trajectories[:, i+1, :N], boundary) #apply periodic boundary conditions
        trajectories[:, i+1, N:] = update_orientation(e, persistence_length, particles_vel, dt=dt, curvity = kappa, Dr = Dr, N = N, mode_particles=mode_particles) #returns arctan2

        #Bookkeeping probabilities 
        if mode == 'Iso_oreo':
            #calculate occupancies repulsive potentials
            for j in range(n_potentials):
                distances = np.linalg.norm(u.pbc_distance(trajectories[j, i+1, :N], trajectories[:, i+1, :N], boundary), axis=1)
                storage_occupancy[j] = len(np.where(distances < (b[j] + 1.1))[0])-1 

    # Calculate filling fraction (outside of repulsive potentials)
    space_repulsive_potentials = np.sum((np.pi**(N/2)/math.gamma(N/2 + 1)) * b[:n_potentials]**N) #hypersphere
    free_box_space = boundary**N - space_repulsive_potentials #space where particles live
    space_particles = np.sum((np.pi**(N/2)/math.gamma(N/2 + 1)) * b[n_potentials:]**N) #hypersphere
    filling_fraction = space_particles/free_box_space 
    
    #saving data
    data_save = 1 #space between stored timesteps
    v0, b, kappa, stiffness = params_particles.T
    params = v_0, curvity, boundary, b, stiffness, t_end, n_particles, persistence_length, radius_repulsive_potential, dt, filling_fraction, int(timestep_end/data_save)
    
    if mode == 'Iso_oreo':
        data = {'Trajectories': trajectories[:, ::data_save], #don't store all data, files get big
                'Params': params,
                'P_b': storage_occupancy[0],
                'P_s': storage_occupancy[1]+storage_occupancy[2]}
    else: 
        data = {'Trajectories': trajectories[:, ::data_save], #don't store all data, files get big
                'Params': params}
    return data

####################################################
# Equations of motion
def update_positions(current_positions, eta, params_particles, params_system, interacting, mode):
    """
    current positions shape (n_particles, N) 
    eta shape (n_particles, N)
    v0, radius particles, shape (n_particles, 1)
    """
    #unpack values for readability
    v0, radius_particles, kappa, stiffness = (x[:, np.newaxis] for x in params_particles.T)
    boundary, t_end, n_particles, persistence_length, radius_repulsive_potential, dt, n_potentials, N = params_system

    #scale mobility with radius
    mobility = v0 * (1/radius_particles) #shape = (n_particles, 1)
    mobility[:n_potentials]=0 #makes repulsive potentials stationary
    
    #Compute forces
    f = 0
    if interacting == True or mode != None:
        f += compute_pairwise_forces(current_positions.shape[0], current_positions, radius_particles, stiffness, boundary, interacting=interacting, N = N, n_potentials=n_potentials, mode=mode) #shape (n_particles, N)  
    return current_positions + dt * (eta * v0) + dt * mobility * f 

def compute_pairwise_forces(n_particles, current_positions, radius_particles, stiffness, boundary, interacting, N, n_potentials, mode):
    """
    n_particles,N: int
    current positions shape (n_particles, N)
    radius_particles, stiffness shape (n_particles, 1)
    """ 
    
    #some bookkeeping
    f = np.zeros((n_particles, N))
    epsilon = 1e-10 #buffer

    # Compute pairwise displacement vectors and apply periodic boundaries    
    r1, r2 = current_positions[:, np.newaxis, :], current_positions[np.newaxis, :, :] #shape (n_particles, 1, N), (1, n_particles, N), for broadcasting
    rij = u.pbc_distance(r1, r2, boundary)  # shape (n_particles, n_particles, N) -> center to center, vector
    
    #absolute distances
    rij_abs = np.linalg.norm(rij, axis=2)  # shape (n_particles, n_particles)
    np.fill_diagonal(rij_abs, np.inf) #to make sure that the force calculated on itself is 0
    
    # Only consider interaction between particles [0,1,2] and [3:N]
    if (mode == 'Iso_oreo') or mode == 'Arrhenius':
        rij_sub = rij[:n_potentials, n_potentials:]    
        rij_abs_sub = rij_abs[:n_potentials, n_potentials:]  
        radii_sum = radius_particles[:n_potentials] + radius_particles[n_potentials:].T  # (n_potentials, n_particles-n_potentials)
        
        #harmonic potential
        gamma = np.where(rij_abs_sub <= radii_sum, (stiffness[:n_potentials]/2)*(radii_sum-rij_abs_sub)**2,0.0)#np.exp(stiffness[:n_potentials] * (1 - rij_abs_sub / radii_sum)), 0.0)  # (n_potentials, n_particles-n_potentials)
        forces = gamma[:, :, np.newaxis] * rij_sub / (rij_abs_sub[:, :, np.newaxis] + epsilon)  # (n_particles, n_particles-n_potentials, N)
        f[:n_potentials] += np.sum(forces, axis=1)  # sum over the second axis 
        f[n_potentials:] -= np.sum(forces.transpose(1, 0, 2), axis=1) #(n_particles-n_potentials, n_potentials, N)
    
    else:
        radii_sum = radius_particles + radius_particles.T # shape (n_particles, n_particles)
        gamma = np.where(rij_abs<2**(1/6)*2, 4 * (((radius_particles*2)/rij_abs)**12-((radius_particles*2)/rij_abs)**6)+1, 0.0) #WCA
        forces = gamma[:, :, np.newaxis] * rij / (rij_abs[:, :, np.newaxis] + epsilon)  # (n_particles, n_particles, N)
        f += np.sum(forces, axis=1)  # sum over the second axis: shape (n_particles, N)
    return f

#Orientation update
def update_orientation(e, persistence_length, particles_vel, dt, curvity, Dr, N, mode_particles):
    """
    shape e: (n_particles, N)
    shape particles_vel (n_particles, N)
    curvity: (n_particles, 1)
    """
    n_particles = e.shape[0]
    
    if N == 2:
        #deterministic part 
        cross = particles_vel[:, 0] * e[:, 1] - particles_vel[:, 1] * e[:, 0]  # shape: (n_particles,), scalars
        change = - cross[:, np.newaxis] * curvity * np.array([-e[:, 1], e[:, 0]]).T # shape: (n_particles, 2)
        e += change * dt  # shape: (n_particles, N)

        #stochastic part
        if mode_particles == 'ABP':
            noise = np.sqrt(N*Dr*dt)*np.random.normal(0, 1, size = (n_particles, 1)) * np.array([-e[:, 1], e[:, 0]]).T 
            e += noise
        elif mode_particles == 'RTP':
            tumble_angle = np.random.uniform(-1, 1, size=(n_particles, N)) 
            #Dr is the tumbling rate alpha, see solon 2016
            poisson = np.random.poisson(Dr, size = (n_particles, N)) 
            e += tumble_angle*poisson * np.sqrt(dt)
            
        norm = np.linalg.norm(e, axis=1) 
        e /= norm[:, np.newaxis]  
        arctan = np.arctan2(e[:, 1], e[:, 0]) #returns (N-1) orientation DOF, arctan (x2,x1)
        dof = arctan[:, np.newaxis]

    elif N == 3:
        #deterministic part
        i_hat = particles_vel[:, 1] * e[:, 2] - particles_vel[:, 2] * e[:, 1] #i_hat
        j_hat = particles_vel[:, 0] * e[:, 2] - particles_vel[:, 2] * e[:, 0]
        k_hat = particles_vel[:, 0] * e[:, 1] - particles_vel[:, 1] * e[:, 0]
        cross_3d_1 = np.stack([i_hat, -j_hat, k_hat], axis = 1) #shape: n_particles, N
        i_hat = e[:, 1] * cross_3d_1[:, 2] - e[:, 2] * cross_3d_1[:, 1] #i_hat
        j_hat = e[:, 0] * cross_3d_1[:, 2] - e[:, 2] * cross_3d_1[:, 0]
        k_hat = e[:, 0] * cross_3d_1[:, 1] - e[:, 1] * cross_3d_1[:, 0]
        cross_3d_2 = np.stack([i_hat, -j_hat, k_hat], axis = 1) #shape: n_particles, N
        change = cross_3d_2 * curvity
        e += change * dt  # shape: (2, n_particles)

        #todo, fix stochastic part
        
        norm = np.linalg.norm(e,axis=1)
        e /= norm[:, np.newaxis] 
        theta = np.arctan2(e[:, 1], e[:, 0]) 
        phi = np.arccos(e[:, 2]) 
        dof = np.stack([theta, phi], axis = 1) #theta, phi
    return dof #shape (n_particles, N-1)


#todo numba, neighbour list, option for RK
