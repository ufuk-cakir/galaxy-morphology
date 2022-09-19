import numpy as np

from swiftsimio.visualisation.projection import scatter as scatter2D
from swiftsimio.visualisation.volume_render import scatter as scatter3D

import plotly.graph_objects as go

from matplotlib import cm
from mayavi import mlab
import matplotlib.pyplot as plt
import os 
from tqdm import trange, tqdm

#------------- Coordinate Rotation -------------
def radial_distance(coords,center):
    d = coords-center
    r_i = d**2
    r= np.sqrt(np.sum(r_i, axis = 1))
    return(r)

def momentOfIntertia(gas, rHalf, subhalo_pos,stars, gas_flag = True): #gas_flag = True if gas particles  exist
    """ Calculate the moment of inertia tensor (3x3 matrix) for a subhalo-scope particle set."""
    #t = PlotGalaxy(halo_id)
    #rHalf = subhalos["SubhaloHalfmassRadType"][4] #4 is PartTypeNum for Stars Particles
    #subhalo_pos = subhalos["SubhaloPos"]

    #gas = il.snapshot.loadSubhalo(t.basePath,t.snapshot_number,t.halo_id, "gas")
    
    rad = radial_distance(coords = stars["Coordinates"], center = subhalo_pos) if not gas_flag else radial_distance(coords = gas["Coordinates"], center = subhalo_pos)
    wGas = np.where((rad <= 2.0*rHalf))[0] if not gas_flag else np.where((rad <= 2.0*rHalf) &(gas["StarFormationRate"]>0.0))[0]
    masses = stars["Masses"][wGas] if not gas_flag else gas["Masses"][wGas]
    xyz = stars["Coordinates"][wGas,:] if not gas_flag else gas["Coordinates"][wGas,:]  
    #Shift
    xyz = np.squeeze(xyz)


    if xyz.ndim == 1:
        xyz = np.reshape( xyz, (1,3) )

    for i in range(3):
        xyz[:,i] -= subhalo_pos[i]

    # if coordinates wrapped box boundary before shift:
    # sP.correctPeriodicDistVecs(xyz) was ist das?

    I = np.zeros( (3,3), dtype='float32' )

    I[0,0] = np.sum( masses * (xyz[:,1]*xyz[:,1] + xyz[:,2]*xyz[:,2]) )
    I[1,1] = np.sum( masses * (xyz[:,0]*xyz[:,0] + xyz[:,2]*xyz[:,2]) )
    I[2,2] = np.sum( masses * (xyz[:,0]*xyz[:,0] + xyz[:,1]*xyz[:,1]) )
    I[0,1] = -1 * np.sum( masses * (xyz[:,0]*xyz[:,1]) )
    I[0,2] = -1 * np.sum( masses * (xyz[:,0]*xyz[:,2]) )
    I[1,2] = -1 * np.sum( masses * (xyz[:,1]*xyz[:,2]) )
    I[1,0] = I[0,1]
    I[2,0] = I[0,2]
    I[2,1] = I[1,2]

    return I



def rotationMatrix(inertiaTenor, return_value = "face-on"):
    """ Calculate 3x3 rotation matrix by a diagonalization of the moment of inertia tensor.
    Note the resultant rotation matrices are hard-coded for projection with axes=[0,1] e.g. along z. """

    # get eigen values and normalized right eigenvectors
    eigen_values, rotation_matrix = np.linalg.eig(inertiaTenor)

    # sort ascending the eigen values
    sort_inds = np.argsort(eigen_values)
    eigen_values = eigen_values[sort_inds]

    # permute the eigenvectors into this order, which is the rotation matrix which orients the
    # principal axes to the cartesian x,y,z axes, such that if axes=[0,1] we have face-on
    new_matrix = np.matrix( (rotation_matrix[:,sort_inds[0]],
                             rotation_matrix[:,sort_inds[1]],
                             rotation_matrix[:,sort_inds[2]]) )

    # make a random edge on view   ---> rotation entlang x achse
    #new_matrix auf eigenvektor, winkel berechnen, rotation matrix bauen
    phi = np.random.uniform(0, 2*np.pi)
    theta = np.pi / 2
    psi = 0

    A_00 =  np.cos(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.sin(psi)
    A_01 =  np.cos(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.sin(psi)
    A_02 =  np.sin(psi)*np.sin(theta)
    A_10 = -np.sin(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.cos(psi)
    A_11 = -np.sin(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.cos(psi)
    A_12 =  np.cos(psi)*np.sin(theta)
    A_20 =  np.sin(theta)*np.sin(phi)
    A_21 = -np.sin(theta)*np.cos(phi)
    A_22 =  np.cos(theta)

    random_edgeon_matrix = np.matrix( ((A_00, A_01, A_02), (A_10, A_11, A_12), (A_20, A_21, A_22)) )

    # prepare return with a few other useful versions of this rotation matrix
    r = {}
    r['face-on'] = new_matrix
    r['edge-on'] = np.matrix( ((1,0,0),(0,0,1),(0,-1,0)) ) * r['face-on'] # disk along x-hat
    r['edge-on-smallest'] = np.matrix( ((0,1,0),(0,0,1),(1,0,0)) ) * r['face-on']
    r['edge-on-y'] = np.matrix( ((0,0,1),(1,0,0),(0,-1,0)) ) * r['face-on'] # disk along y-hat
    r['edge-on-random'] = random_edgeon_matrix * r['face-on']
    r['phi'] = phi
    

    return r[return_value]





#------------- IMAGE GENERATION -------------

def normalize(arr):
    arr_min = np.min(arr[np.isfinite(arr)])
    scale_arr = np.zeros_like(arr)
    scale_arr[np.isfinite(arr)] = (arr[np.isfinite(arr)]-arr_min)/(np.max(arr[np.isfinite(arr)])-arr_min)
    return scale_arr




def image2D(coordinates, R_half, weights, smoothing_length, normed = False, plot_factor = 1.5, res = 128):
    
    plot_range = plot_factor*R_half
    
    x = coordinates[:,0].copy()
    y = coordinates[:,1].copy()
    
    m =  weights
    
    h = smoothing_length.copy()
    
    #Transform Particles s.t -factor*r_halfmassrad < x <factor*r_halfmassrad -> 0 < x <1
    x = x/(2*plot_range) +1/2  
    y = y/(2*plot_range) +1/2

    h = h/(2*plot_range)
    
    SPH_hist = scatter2D(x=x, y = y,h = h, m = m ,res= res)
    if normed:
        SPH_hist = SPH_hist-SPH_hist.min()
        SPH_hist = SPH_hist/SPH_hist.max()
        
    return(SPH_hist)



def plot3d(hist3d, min_val = 1e-5, marker_size = 25, colorscale = "viridis", opacity = 0.6):
    data_hist = hist3d.copy()
    data_hist[np.where(data_hist <min_val)] = 0
    xx, yy, zz = np.where(data_hist != 0)
    s = data_hist[xx,yy,zz]
    fig = go.Figure(data = [go.Scatter3d(
    x = xx,
    y=yy,
    z=zz,
    mode = "markers",
    marker = dict(size=marker_size*s, color = s, colorscale=colorscale, opacity = opacity)  )])
    fig.show()

def image3D(coordinates, R_half, weights, smoothing_length,show_plot = True, normed = False, plot_factor = 1.5, res = 128,
            min_val = 1e-5, marker_size = 25, colorscale = "viridis", opacity = 0.6, log = False):
    
    #part = self.vertical_rotated_part()
    plot_range = plot_factor*R_half
    
    x = coordinates[:,0].copy()
    y = coordinates[:,1].copy()
    z = coordinates[:,2].copy()
    
    #m = np.log10(self.particle_data["Masses"]*1e10)
    m =  weights
    
    h = smoothing_length.copy()
    
    #Transform Particles s.t -factor*r_halfmassrad < x <factor*r_halfmassrad -> 0 < x <1
    x = x/(2*plot_range) +1/2  
    y = y/(2*plot_range) +1/2
    z = z/(2*plot_range) +1/2

    h = h/(2*plot_range)
    
    SPH_hist = scatter3D(x=x, y = y, z = z,h = h, m = m ,res= res)
    
    #SPH_hist = (SPH_hist -1/2)*(2*plot_range)
    

    
    
    if log: SPH_hist = np.log10(SPH_hist)
    if normed: SPH_hist=normalize(SPH_hist)
    if show_plot: plot3d(SPH_hist,min_val = min_val, marker_size = marker_size, colorscale = colorscale, opacity = opacity )
        
        
    return(SPH_hist)


#------------- Visualization -------------
def mayaviContour(volume, colorbar = False, azimuth = 90, elevation = -50, distance = 180, contour_count = 500, opacity = .01 , save_path = None):
    fig1 = mlab.figure(size=(1000, 1000), bgcolor=(0, 0, 0), fgcolor=(1, 1,1))
    scalars = mlab.pipeline.scalar_field(volume)
    contours = mlab.pipeline.contour_surface(scalars,contours = contour_count,transparent=True, opacity = opacity,colormap = "magma")
    if colorbar: mlab.colorbar(orientation="vertical", nb_labels=3)
    mlab.view(azimuth=azimuth, elevation=elevation, distance=distance)
    if save_path is not None:
        mlab.savefig(save_path,figure = fig1)
    mlab.show()
    

def mayaviScatter(hist, min_val = 1e-5, save_name = None,show_plot = True):
    cmap = plt.get_cmap('viridis')
    cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255
    data = hist.copy()
    data[np.where(data <min_val)] = 0

    xx, yy, zz = np.where(data != 0)

    s = data[xx,yy,zz]



    p = mlab.points3d(xx, yy, zz,s, colormap = "copper",scale_factor=1, mode = "cube")
    p.module_manager.scalar_lut_manager.lut.table = cmaplist
    #p.glyph.scale_mode = "scale_by_vector"
    if save_name is not None:
        os.makedirs("data/mayavi", exist_ok=True)
        mlab.savefig(filename=f'data/mayavi/{save_name}.png')
    if show_plot: mlab.show()


def volume(hist ,opacity = .1, isomin = 0, isomax = None, surface_count = 30):
    if isomax is None: isomax = hist.max()
    data_hist =hist.copy()
    xx, yy, zz = np.where(data_hist != 0)
    s = data_hist[xx,yy,zz]
    fig = go.Figure(data=go.Volume(
        x=xx,
        y=yy,
        z=zz,
        value=s,
        isomin=isomin,
        isomax=isomax,
        opacity=opacity, # needs to be small to see through all surfaces
        surface_count=surface_count,# needs to be a large number for good volume rendering
        ))
    fig.update_layout(scene_xaxis_showticklabels=False,
                  scene_yaxis_showticklabels=False,
                  scene_zaxis_showticklabels=False)
    fig.show()



def plot_grid(data, save_name = "None", cmap =  cm.magma, dpi = 100):
    """Plot images in a grid. Gridsize will be calculated from length of inputarray.

    :param data:Input images to be plotted
    :type data: list
    :param save_name: Name of output file, defaults to "None"
    :type save_name: str, optional
    :param cmap: Colormap of images, will be passed to matplotlib.pyplot.imshow, defaults to cm.magma
    :type cmap: _type_, colormap
    :param dpi: dpi of output image, defaults to 100
    :type dpi: int, optional
    """    
    x = int(np.floor(np.sqrt(len(data))))
    y = int(len(data)/x)
    
    fig, ax = plt.subplots(x,y, figsize = (50,50))
    k = 0
    for i in trange(x):
        for j in range(y):
            ax[i,j].imshow(data[k], cmap = cmap)
            ax[i,j].set_title(k)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            k = k+1
        

    #plt.show()
    if save_name!="None":
        fig.savefig(f"{save_name}_grid.png", dpi = dpi)


#------------- Image Processing -------------

def clip_hist(data, lower = 0.1, upper = 1.):
    """Clip image to [lower,upper] quantile.

    :param data: Input image
    :type data: 2D numpy array
    :param lower: lower qunatile, defaults to 0.1
    :type lower: float, optional
    :param upper: upper qunatile, defaults to 1.
    :type upper: float, optional
    """    
    hist = data.copy()
    L,U = np.quantile(hist,[lower,upper])
    hist = np.clip(hist, L, U)
    return(hist)


def norm_hist(hist_raw):
    hist = hist_raw.copy()
    hist = hist - hist.min()
    if hist.max() == 0: 
        print("Maximum is 0.")
        return hist 
    else: 
        return(hist/hist.max())

def log(hist_raw):
    hist = hist_raw.copy()
    hist[np.where(hist == 0)] = 1
    hist = np.log10(hist)
    return(hist)


    