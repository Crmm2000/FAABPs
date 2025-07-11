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
        i_hat = v_1[:, 1] * v_2[:, 2] - v_1[:, 2] * v_2[:, 1] 
        j_hat = v_1[:, 0] * v_2[:, 2] - v_1[:, 2] * v_2[:, 0]
        k_hat = v_1[:, 0] * v_2[:, 1] - v_1[:, 1] * v_2[:, 0]
        cross = np.stack([i_hat, -j_hat, k_hat], axis = 1) #shape: n_particles, N
    return cross

#neighbour list
def neighbour_list_compute_pairwise_forces(n_particles, current_positions, radius_particles, stiffness, boundary, interacting, N):
    """
    n_particles,N: int
    current positions shape (n_particles, N)
    radius_particles, stiffness shape (n_particles, 1)
    """ 
    #some bookkeeping 
    f = np.zeros((n_particles, N))
    epsilon = 1e-10 #buffer

    #neighbour listing
    head, list_next, n_cells, cell_size = create_cell_list(current_positions, radius_particles, boundary)

    # check the cell a particles belongs to
    #cell_coords = (current_positions / cell_size).astype(int) #shape (n_particles, N)
    for i in range(n_particles):
        #from eduardo
        cell_x, cell_y = int(current_positions[i, 0] / cell_size), int(current_positions[i, 1] / cell_size)
        if N == 3:
            cell_z = int(current_positions[i, 2] / cell_size)
        #coord = tuple(cell_coords[i])
        #coords_grid = np.ones((3,)* N + (N,)) * coord # shape (3, 3, 2) in 2D
        # gradient = np.arange(-1, 2)
        # gradients = np.meshgrid(*([gradient] * N)) # extends to multiple dimensions, shape (N,3,3,...)
        # gradients = np.stack(gradients, axis=-1) #shape (3,3,...,N)
        # neighbours_coords = coords_grid + (gradients)
        # neighbours_list = neighbours_coords.reshape(-1, 2) 
        # neighbours_list = neighbours_list % n_cells #pbc
        #print(f'coord: {coords_grid}, neighbours: {neighbours_coords}')
        if N == 2:
            for dx in range(-1,2):
                for dy in range(-1,2):
                #for k in range(9):
                    #neighbours_cell = tuple(neighbours_list[i])
                    #neighbours_cell = tuple(int(x) for x in neighbours_list[k])
                    neigh_x = (cell_x + dx) % n_cells
                    neigh_y = (cell_y + dy) % n_cells
                    j = head[neigh_x, neigh_y] #int(head[neighbours_cell])
                    while j != -1:
                        if i != j:
                            r1, r2 = current_positions[i], current_positions[j] 
                            rij = pbc_distance(r1, r2, boundary)  
                            rij_abs = np.linalg.norm(rij)  
                            radii_sum = radius_particles[i] + radius_particles[j] 
                            #WCA
                            if rij_abs<2**(1/6)*2:
                                #gamma = 4 * (((radius_particles[j]*2)/rij_abs)**12-((radius_particles[j]*2)/rij_abs)**6)+1 
                                gamma = 4 * (((2)/rij_abs)**12-((2)/rij_abs)**6)+1 
                            #gamma = np.exp(stiffness[j] * (1 - rij_abs / radii_sum))
                            else: gamma = 0
                            forces = gamma * rij / (rij_abs + epsilon) 
                            f[i] += forces 
                        j = list_next[j]
        elif N == 3:
            for dx in range(-1,2):
                for dy in range(-1,2):
                    for dz in range(-1,2):
                #for k in range(9):
                    #neighbours_cell = tuple(neighbours_list[i])
                    #neighbours_cell = tuple(int(x) for x in neighbours_list[k])
                        neigh_x = (cell_x + dx) % n_cells
                        neigh_y = (cell_y + dy) % n_cells
                        neigh_z = (cell_z + dz) % n_cells
                        j = head[neigh_x, neigh_y, neigh_z] #int(head[neighbours_cell])
                        while j != -1:
                            if i != j:
                                r1, r2 = current_positions[i], current_positions[j] 
                                rij = pbc_distance(r1, r2, boundary)  
                                rij_abs = np.linalg.norm(rij)  
                                radii_sum = radius_particles[i] + radius_particles[j] 
                                #WCA
                                if rij_abs<2**(1/6)*2:
                                    #gamma = 4 * (((radius_particles[j]*2)/rij_abs)**12-((radius_particles[j]*2)/rij_abs)**6)+1 
                                    gamma = 4 * (((2)/rij_abs)**12-((2)/rij_abs)**6)+1 
                                #gamma = np.exp(stiffness[j] * (1 - rij_abs / radii_sum))
                                else: gamma = 0
                                forces = gamma * rij / (rij_abs + epsilon) 
                                f[i] += forces 
                            j = list_next[j]
    return f

#neighbour list
def create_cell_list(current_positions, radius_particles, L):
    """
    current positions shape (n_particles, N) 
    eta shape (n_particles, N)
    v0, radius particles, shape (n_particles, 1)
    """

    N, n_particles = current_positions.shape[1], current_positions.shape[0]  
    cell_size = 2 * np.max(radius_particles) #see whether this can be more specific 
    n_cells = int(np.floor(L / cell_size))

    head = np.full(((n_cells,) * N), -1, dtype=int) # (n_cells * n_cells) for 2D
    list_next = np.full(n_particles, -1, dtype=int)

    cell_coords = (current_positions / cell_size).astype(int) #shape (n_particles, N)

    for i in range(n_particles): #for loop is needed for list -> check whether this can be different
        coord = tuple(cell_coords[i])
        list_next[i] = head[coord]
        head[coord] = i

    return head, list_next, n_cells, cell_size

# add gradients for potentials

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
    n = int(t_end/dt) #timesteps
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
            for i in range(n_potentials):
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
    
    elif mode == 'Arrhenius':
        # add a single repulsive potential
        v0[0], b[0], kappa[0], stiffness[0], n_potentials = 0, radius_repulsive_potential, 0, stiffness_potential, 1

        #place particles near the potential
        epsilon = 0.1 #buffer
        x_0 = [boundary/2] * N
        x_0[0] += radius_repulsive_potential + epsilon #np.random.uniform(0, boundary, shape=(n_particles, N))
        trajectories[:, 0, :N] = x_0
        trajectories[:, 0, N] = np.pi
        trajectories[0, 0, :N] = [boundary/2] * N #place potential in the centre

        #simulation
        terminate = False #set termination condition to finish simulation once all particles left

        #storage
        particles_within_range = np.zeros(n) #tracks how many particles are near the potential at each timestep
        particles_still_trapped = np.ones(n) * (n_particles-1) # N(t), tracks the amount of particles that are trapped

        #put values back
        params_system = boundary, t_end, n_particles, persistence_length, radius_repulsive_potential, stiffness_potential, dt, n_potentials, N
        params_particles = np.column_stack([v0, b, kappa, stiffness])
        n_escape = np.zeros(n_particles) #tracks particles that are out of range
        return particles_within_range, particles_still_trapped, n_escape, params_particles, params_system, trajectories, terminate
