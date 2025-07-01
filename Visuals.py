import numpy as np
import time #tracks time

# packages for visualisation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from IPython import display
import imageio.v2 as imageio

# creates the video using the previous render function
def save_fig(position, orientation, params, steps_per_frame, fps, name_file, title = '', N = 2, live = False, view = None):
    n = params[11] #amount of timesteps
    frames = [] #create empty list to store frames as video
    for i in range(n): #loop over the time steps
        if i % steps_per_frame == 0: #extra condition to speed up loop to not visualize every time step
            progress = ((i) * (params[9])) #progressbar
            fig = render(position[:, i, :], orientation[i, :], f"$t$ = {progress:.0f} $s$, {title}", params=params, N = N, live = live, view = view)
            plt.savefig('temp_plot.png', bbox_inches='tight')
            frames.append(imageio.imread('temp_plot.png'))
            plt.close(fig)
    imageio.mimsave(f"{name_file}.mp4", frames, fps = fps) #gif instead of mp4, just change .mp4 to .gif, but mind that fps means something different

#Render figures for videos
def render(states, orientation, progress, params, N = 2, live = False, view = None):
    boundary = params[2]
    size_particles = params[3]
    if N == 2:
        #plotting formalities
        plt.figure()
        plt.title(f'$n$ = {states.shape[0]}, $v_0$ = {params[0]}, $l_0$ = {params[7]}, $\\kappa$ = {params[1]}, $\\phi = {params[10]:.2f}$')
        plt.suptitle(f'{progress}')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.xlim([0,boundary])
        plt.ylim([0, boundary])

        # coloring according to the orientation 
        orientation = np.mod(orientation, 2*np.pi)  # keeps values on circle
        orientation[orientation < 0] += 2*np.pi  # keeps values positive
        norm = mcolors.Normalize(vmin=0, vmax=2*np.pi) #normalizing for colormap
        cmap = cm.hsv  #set colormap -> hsv is cyclic and allows for a proper separatin of colors irt the background

        #plotting
        if len(states.shape) == 1: #extra line for if there is only one particles
            plt.scatter(states[0], states[1], color=cmap(norm(orientation)), s=10)
        else: #normal setting hwne there are multiple particles
            sc = plt.scatter(states[:3, 0], states[:3, 1], color='k', s=(size_particles[:3]**2.1 * 2000 / ((0.1 * boundary)**2)), alpha=0.8, edgecolors='k')

            # Plot the rest of the particles with orientation-based colormap
            sc = plt.scatter(states[3:, 0], states[3:, 1], c=orientation[3:], cmap=cmap, norm=norm,
                            s=(size_particles[3:]**2.2 * 2000 / ((0.1 * boundary)**2)), alpha=0.8)
        # settings for the colorbar for the orientation
        cbar = plt.colorbar(sc, location='right')  
        cbar.set_ticks(ticks=[0,0.5*np.pi,np.pi, 1.5*np.pi, 2*np.pi], labels=['0','$\\frac{\\pi}{2}$', '$\\pi$', '$\\frac{3\\pi}{2}$' , '$2\\pi$'])
        cbar.set_label("$\\hat{e}$")
        if live == True:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        
    elif N == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim([0, boundary])
        ax.set_xlim([0, boundary])
        ax.set_zlim([0, boundary])
        ax.set_box_aspect([1, 1, 1])
        plt.title(f'$n$ = {states.shape[0]}, $v_0$ = {params[0]}, $l_0$ = {params[7]}, $\\kappa$ = {params[1]}, $\\phi = {params[10]:.2f}$')
        plt.suptitle(f'{progress}')
        sizes = (size_particles ** 2.2) * 2000 / ((0.1 * boundary) ** 2)
        sc = ax.scatter(states[3:, 0], states[3:, 1], states[3:, 2],
                    c='r', s=sizes[3:], alpha=0.8, edgecolors='k')
        sc = ax.scatter(states[:3, 0], states[:3, 1], states[:3, 2],
                    c='k', s=sizes[:3], alpha=0.8)
        if view == 'x':
            ax.view_init(elev=0, azim=0)
        elif view == 'y':
            ax.view_init(elev=0, azim=90)
        elif view == 'z':
            ax.view_init(elev = 90, azim = 0)
        if live == True:
            display.display(plt.gcf())
            display.clear_output(wait=True)
