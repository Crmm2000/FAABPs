import numpy as np
import sys

# Periodic boundary conditions
def pbc_vec(states, boundary):
    # makes particle show up on other side
    states[:] = np.mod(states[:], boundary)
    return states

def pbc_distance(r_i, r_j, L):
    # calculates distances across boundary
    r_ij = r_i - r_j 
    r_ij = r_ij - L * np.round(r_ij / L)
    return r_ij

def cross_product(v_1, v_2, N):
    'shape has to be (n_particles, N)'
    if N == 2:
        cross = v_1[:, 0] * v_2[:, 1] - v_1[:, 1] * v_2[:, 0]
    elif N == 3:
        i_hat = v_1[:, 1] * v_2[:, 2] - v_1[:, 2] * v_2[:, 1] #i_hat
        j_hat = v_1[:, 0] * v_2[:, 2] - v_1[:, 2] * v_2[:, 0]
        k_hat = v_1[:, 0] * v_2[:, 1] - v_1[:, 1] * v_2[:, 0]
        cross = np.stack([i_hat, -j_hat, k_hat], axis = 1) #shape: n_particles, N
    return cross

# Experiments
def setup_experiment(mode, params_particles, params_system, trajectories):
    '''
    Gives the main necessary storage arrays for the experiments  
    sets initial parameters repulsive potentials and particles

    params_particles (n_particles, 4), columns are self propulsion speed, radius, curvity, stiffness
    params_system = boundary, t_end, n_particles, persistence_length, radius_repulsive_potential, dt, n_potentials
    '''  
    #unpack values, for readability
    boundary, t_end, n_particles, persistence_length, radius_repulsive_potential, stiffness_potential, dt, n_potentials, N = params_system
    v0, b, kappa, stiffness = params_particles.T

    if mode == 'Iso_oreo':
        #create three repulsive potentials
        for i in range(3):
            v0[i], b[i], kappa[i], stiffness[i], n_potentials = 0, radius_repulsive_potential, 0, stiffness_potential, 3
            if i == 2:
                b[i] = 2*radius_repulsive_potential
        
        # Places repulsive potentials at an appropriate distance + small particles don't spawn in it
        max_iter, iter = 100000, 0
        overlap = True
        while overlap == True:
            overlap = False
            for i in range(n_particles):
                for j in np.arange(i+1, n_particles):
                    distance = np.linalg.norm(pbc_distance(trajectories[i,0,:2], trajectories[j,0,:2], boundary))
                    if j > 2: epsilon = 0.1
                    else: epsilon = 22 + radius_repulsive_potential #persistence_length*2 + R
                    if distance <= b[i] + b[j] + epsilon :
                        overlap = True
                        x_0 = np.random.uniform(0, boundary, size=(N))
                        trajectories[j, 0, :N] = x_0
                        iter += 1
                        if iter == max_iter: print('Try a smaller repulsive potential: Max iter for placing particles reached'); sys.exit()   
        #put values back
        params_system = boundary, t_end, n_particles, persistence_length, radius_repulsive_potential, stiffness_potential, dt, n_potentials, N
        params_particles = np.column_stack([v0, b, kappa, stiffness])
        return np.zeros((n_potentials, int(t_end/dt))), params_particles, params_system
