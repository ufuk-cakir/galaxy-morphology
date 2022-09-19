
import visualization as viz


import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from tqdm import tqdm
from tqdm import trange

import seaborn as sns; sns.set()
from glob import glob
from matplotlib import cm
import imageio.v2 as imageio


import h5py

sns.set_context("notebook") 
sns.set_style("darkgrid", {'axes.grid' : False})

plt.rcParams['figure.figsize'] = (10, 10)



hdf5_path = "thesisData.hdf5" #Path to HDF5 File generated from illustrisData


outlier = np.load("outlier_index.npy")



import os
import random
class IllustrisData():
    def __init__(self, hdf_path,  mass_min = 9.5, mass_max = 11.5):
        self.hdf_path = hdf_path
        
        
        self.particle_type = "stars" #das hier noch aus dem hdf5 file nehmen
        self.weights = "Masses"
        self.snapshot_number = 99
        
        
        
     
        with h5py.File(self.hdf_path,mode ="r") as f:
        #self.hdf = h5py.File(self.hdf_path,mode ="r")
        # hier vllt eine methode, es nicht direkt zu öffnen sonder on disk zu lassen, und später drauf zugreife
            self.mass = f["TNG"]["Mass"][()]
            self.mask =np.where((self.mass> mass_min) &(self.mass< mass_max))[0]
            self.mask = np.setdiff1d(self.mask, outlier)
            self.images = f["TNG"]["image"][self.mask]
            self.mass = self.mass[self.mask]
            self.halo_id = f["TNG"]["halo_id"][self.mask]
            self.p_late = f["TNG"]["P_late"][self.mask].flatten() 
            self.images_shape = self.images.shape #das hier brauch ich nicht
        #self.p_late = self.hdf["TNG"]["P_late"]
        #self.sigma_late = self.hdf["TNG"]["sigma_late"
        self.plots_path = "data/plots"
        self.eigengalaxies_path = "data/eigengalaxies"
        
        
    
        
        os.makedirs(self.plots_path,exist_ok=True)
        os.makedirs(self.eigengalaxies_path,exist_ok=True)
    
    
    
    def norm(self, hist, log10 = False, normed = False, clip = False, lower = .1):
        if log10: hist = viz.log(hist)
        if clip: hist = viz.clip_hist(hist, lower = lower, upper =1)
        if normed: hist = viz.norm_hist(hist)
        
        return(hist)
        
    
    def show_galaxy(self,index, cmap = cm.magma, normed = False, clip = False, lower = .1, log10 = False):
        
        fig = plt.figure(figsize = (10,10))
        hist = self.images[index]
        hist = self.norm(hist, clip = clip, lower = lower, normed= normed, log10 = log10)
        mass = round(self.mass[index]) #warum geht das nicht???!?!?!?!?!?!?!?

        plt.imshow(hist, cmap = cmap,extent=[-hist.shape[1]/2., hist.shape[1]/2., -hist.shape[0]/2., hist.shape[0]/2. ])
        plt.colorbar()
        plt.title(f"Histogram of {self.particle_type} particles")
        plt.figtext(0.1,0.9, f"halo_id ={self.halo_id[index]}")
        plt.figtext(0.1,.88, f"snapshot_number ={self.snapshot_number}")
        plt.figtext(0.1,0.86, f"Mass ={mass}*10^10 M_sun")
        plt.figtext(0.1,.84, f"weights ={self.weights}")
        return(plt)        
    
    def calc_pca(self, n_components = 20, mass_min =9.5, mass_max = 11.5, cmap =  cm.magma, normed = False, clip = False, lower = .1, log10= False):
        print("Reshape Data...")
        self.clipped = True if clip else False

        self.images = self.norm(self.images, log10 = log10, normed = normed, clip = clip,lower = lower)
        pca_data = self.images[()].reshape(self.images_shape[0],self.images_shape[1]**2)
        
        
        self.pca = PCA(n_components=n_components)
        print("Fit to data..")
        self.pca.fit(pca_data)
        print("Transform..")
        self.scores = self.pca.transform(pca_data)
        self.eigengalaxies = self.pca.components_.reshape(n_components,self.images_shape[1],self.images_shape[2])
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.explained_variance = self.explained_variance_ratio.sum()
        pca_inverse = self.pca.inverse_transform(self.scores)
        self.pca_images = pca_inverse.reshape(self.images_shape[0],self.images_shape[1],self.images_shape[2])
        self.pca_likelihood = self.pca.score_samples(pca_data)
        print("Finished! :)")
        print("_________________________________")
        print("Explained Variance:", self.explained_variance)
        print("_________________________________")
        plt.figure(figsize = (15,10))
        plt.scatter(np.arange(n_components), np.cumsum(self.explained_variance_ratio))
        plt.title("Explained Variance Ratio vs. Number Of Principal Components")
        del pca_data
        
    
        
        del pca_data
    def show_eigengalaxies(self,cmap =  cm.magma):
        viz.plot_grid(self.eigengalaxies,save_name=f"{self.eigengalaxies_path}/eigengalaxies+.png", cmap = cmap)
      
    def compare_pca_results(self,index = None, cmap =  cm.magma):
        if index is None: index = random.choice(range(self.images_shape[0]))
        
        
        fig, ax =plt.subplots(1,3, figsize= (20,20))
        im1=ax[0].imshow(self.images[index],cmap = cmap)
        im2=ax[1].imshow(self.pca_images[index],cmap = cmap)
        ax[0].set_title(f"Original:  {self.halo_id[index]}")
        ax[1].set_title("PCA Reconstruction")
        residue = (self.images[index]-self.pca_images[index])/self.images[index] 
        im3=ax[2].imshow(residue,cmap = cm.coolwarm,vmin = -1, vmax=1)
        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].set_title("Residue")
        #divider = make_axes_locatable(ax[0])
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        #fig.colorbar(im=im1,cax = cax)
        plt.show()


    def create_reconstruction_gif(self,galaxy_id, remove = False):
        path = f"animations/footage/{galaxy_id}"
        os.makedirs(path, exist_ok=True)
        reconstructed = self.pca.mean_.reshape(64,64).copy()
        fig, ax = plt.subplots(1,3, figsize = (15,15))
        ax[0].imshow(self.images[galaxy_id], cmap = cm.magma)
        ax[0].set_title("Original")
        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].axis("off")
        for i in trange(len(self.eigengalaxies)):
            ax[1].imshow(reconstructed, cmap = cm.magma)
            ax[1].set_title(f"number of eigengalaxies: {i+1}")
            residue = (self.images[galaxy_id]- reconstructed)/self.images[galaxy_id]
            ax[2].imshow(residue,vmin = -1, vmax = 1, cmap = cm.coolwarm)
            ax[2].set_title("residue")
            fig.savefig(f"{path}/reconstruc_{i}.png")
            plt.close()
            reconstructed += self.scores[galaxy_id][i]*self.eigengalaxies[i]
            
        print("creating gif..")
        
        anim_file = f'animations/reconstructed_{galaxy_id}.gif'
        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob(f'{path}/reconstruc_*.png')
            filenames= sorted(filenames, key = len)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            
        if remove: _ = [os.remove(filenames) for filenames in filenames]
        
    def reconstuction_error_single(self,index):
        mask = np.where(self.images[index]>0)
        diff = np.abs(self.images[index][mask]-self.pca_images[index][mask])/self.images[index][mask]
        rec_median = np.median(diff)
        return(rec_median)
    
    def reconstuction_error(self): 
        rec_err = np.array([self.reconstuction_error_single(i) for i in range(self.images_shape[0])])
        return(rec_err)
