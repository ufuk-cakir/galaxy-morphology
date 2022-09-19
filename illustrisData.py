import h5py
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import os
import visualization as viz

import illustris_python as il
from sklearn.decomposition import PCA
from tqdm import tqdm





with h5py.File("morphologies_deeplearn.hdf5","r") as f:
    deeplearn_ids = f["Snapshot_99"]["SubhaloID"][()]

DATAPATH = "./data"






class illustrisData():
    
    def __init__(self,halo_id, particle_type = "stars",simulation = "TNG100-1",snapshot = 99):
        
        self.basePath = f"/home/tnguser/sims.TNG/{simulation}/output/"
        self.halo_id = halo_id
        self.particle_type = "stars"
        self.snapshot_number = snapshot
        
        self.rotated_flag = 0
        self.gas_flag=False
        self.gas = 0
        
        self.PartTypeDict = {"gas": 0, "dm": 1, "Tracers": 3, "stars": 4 , "bh": 5}
        
        self.load_data(self.halo_id)
        
        
        
    def load_data(self, halo_id):
        
        self.get_subhalo(halo_id = halo_id)
        self.get_particle_data(halo_id = halo_id)
         
          
    def get_subhalo(self, halo_id):
        
        self.subhalo = il.groupcat.loadSingle(self.basePath, self.snapshot_number,subhaloID =halo_id)
        
        
        self.center = self.subhalo["SubhaloPos"]
        self.mass = self.subhalo["SubhaloMassType"][self.PartTypeDict[self.particle_type]]*1e10
        self.halfmassrad = self.subhalo["SubhaloHalfmassRadType"][self.PartTypeDict[self.particle_type]]
    
    
    
    
    #data Path als Variable später
       
    def get_particle_data(self,halo_id):
        print("loading data.")
        self.stars = il.snapshot.loadSubhalo(self.basePath, self.snapshot_number, self.halo_id, "stars")
        
        #self.gas = il.snapshot.loadSubhalo(self.basePath, self.snapshot_number, self.halo_id, "gas") SOME GALAXIES HAVE TO MUCH GAS PARTICLES, PROCESS GETS KILLED
   
        print("finished.")
         
    def face_on(self):
        # Das hier kann man später verallgemeinern
        if self.rotated_flag: return self.particles
        
        self.rotated_flag = 1
        inertia_tensor = viz.momentOfIntertia(stars = self.stars, gas = self.gas, rHalf=self.halfmassrad, subhalo_pos= self.center, gas_flag = self.gas_flag)
        rotation_matrix = viz.rotationMatrix(inertiaTenor=inertia_tensor, return_value = "face-on")
        
        pos = self.stars["Coordinates"]- self.center
        rot_pos =np.dot(rotation_matrix, pos.transpose()).transpose()
        self.particles = np.asarray(rot_pos)
        return(self.particles)

    
    def calc_rot_mat_around_z(self,angle):
        '''
        Calculates Rotation Matrix around z-axis, with input angle in rad.
        '''
        rot_mat = np.array([[np.cos(angle),-np.sin(angle),0],
                           [np.sin(angle), np.cos(angle),0],
                           [0 , 0, 1]])
        return(rot_mat)
    
    def get_angle(self,hist):
        #angle in rad
        fit=PCA(n_components=2).fit(np.argwhere(hist>=np.quantile(hist,.75)))
        return np.arctan2(*fit.components_[0])
    
    
    def vertical(self, hist):
        img = hist.copy()
        img = viz.clip_noise(img, lower =.9, upper = 1)
        angle = self.get_angle(img)
        vertical_rotation_matrix = self.calc_rot_mat_around_z(-angle)
        position = self.particles.copy()
        vertical_postion =np.dot(vertical_rotation_matrix, position.transpose()).transpose()
        self.particles = vertical_postion
        
        
    
    def image2D(self, weights = "Masses", plot_factor = 5, res = 64, normed = False, return_hist = False, show_plot = False):
        if not self.rotated_flag: self.face_on()
        img = viz.image2D(coordinates = self.particles, R_half = self.halfmassrad, 
                            weights=self.stars[weights], smoothing_length=self.stars["StellarHsml"], normed = normed,plot_factor=plot_factor, res =res)
        
        
        self.vertical(img)
        
        img = viz.image2D(coordinates = self.particles, R_half = self.halfmassrad, 
                            weights=self.stars[weights], smoothing_length=self.stars["StellarHsml"], normed = normed,plot_factor=plot_factor, res =res)
        
        
        if show_plot: plt.imshow(img)
        if return_hist: return(img)
    
    def image3D(self, weights = "Masses", plot_factor = 5, res = 64, normed = True, log = True, return_hist = False,
                min_val = 1e-5, marker_size = 25, colorscale = "viridis", opacity = 0.6,show_plot = True):
        if not self.rotated_flag: self.face_on()
        img = viz.image3D(coordinates = self.particles, R_half = self.halfmassrad, 
                            weights=self.stars[weights], smoothing_length=self.stars["StellarHsml"], normed = normed,plot_factor=plot_factor, 
                                res =res,min_val = min_val, marker_size = marker_size, colorscale = colorscale, opacity = opacity, log = log, show_plot=show_plot)
        
        
        if return_hist: return(img)
        
        
        
        


        
def loadData(halo_ids = deeplearn_ids[1:]):
    datapath = "data"
    
    os.makedirs(datapath, exist_ok = True)
    for halo_id in tqdm(halo_ids):
        #if os.path.exists(f"{datapath}/{halo_id}.npy"): continue
        data = dict()
        galaxy = illustrisData(halo_id)
        data["halo_id"]=halo_id
        data["image"]=galaxy.image2D(show_plot=False, normed = True, return_hist = True)
        data["Mass"]=np.log10(galaxy.mass)
        np.save(f"{datapath}/{halo_id}.npy", data)
        del galaxy
        os.system("clear")
     
    print("Create HDF5 File")
    data = dict(halo_id =[], image = [], Mass = [])
    for halo_id in tqdm(halo_ids):
        dat = np.load(f"{datapath}/{halo_id}.npy", allow_pickle = True).item()
        data["halo_id"].append(dat["halo_id"])
        data["image"].append(dat["image"])
        data["Mass"].append(dat["Mass"])
    file = h5py.File("thesisData.hdf5", "w")
    tng = file.create_group("TNG")
    halo_id = tng.create_dataset("halo_id", data = data["halo_id"], compression = "gzip")
    hist = tng.create_dataset("image", data = data["image"], compression = "gzip")
    Mass = tng.create_dataset("Mass", data = data["Mass"], compression = "gzip")
    file.close()
    print("Finished!")
    
    
    
    
def loadData3D(halo_ids = deeplearn_ids[1:]):
    datapath = "data3D"
    
    os.makedirs(datapath, exist_ok = True)
    for halo_id in tqdm(halo_ids):
        #if os.path.exists(f"{datapath}/{halo_id}.npy"): continue
        data = dict()
        galaxy = illustrisData(halo_id)
        data["halo_id"]=halo_id
        data["image3D"]=galaxy.image3D(show_plot=False, normed = False, log = False,return_hist = True).flatten()
        data["Mass"]=np.log10(galaxy.mass)
        np.save(f"{datapath}/{halo_id}.npy", data)
        del galaxy
        os.system("clear")
     
    print("Create HDF5 File")
    data = dict(halo_id =[], image = [], Mass = [])
    for halo_id in tqdm(halo_ids):
        dat = np.load(f"{datapath}/{halo_id}.npy", allow_pickle = True).item()
        data["halo_id"].append(dat["halo_id"])
        data["image"].append(dat["image"])
        data["Mass"].append(dat["Mass"])
    file = h5py.File("thesisData3D.hdf5", "w")
    tng = file.create_group("TNG")
    halo_id = tng.create_dataset("halo_id", data = data["halo_id"], compression = "gzip")
    hist = tng.create_dataset("image3D", data = data["image3D"], compression = "gzip")
    Mass = tng.create_dataset("Mass", data = data["Mass"], compression = "gzip")
    file.close()
    print("Finished!")
    
    
    
    