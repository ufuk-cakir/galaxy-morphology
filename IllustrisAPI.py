
import requests
import h5py
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import sys
import os
import visualization as viz


DATAPATH = "./data"
baseURL = "http://www.tng-project.org/api/"


class illustrisAPI():
    #at this point, does not load gas particles. some galaxies have too much gas particles, and the process gets killed
    def __init__(self,api_key, halo_id,particle_type = "stars",simulation = "TNG100-1",snapshot = 99,):
        self.headers = {"api-key":api_key}
        self.halo_id = halo_id
        self.particle_type = "stars"
        self.snapshot = snapshot
        self.simulation = simulation
        
        
        
        self.rotated_flag = 0
        self.gas_flag=False
        self.gas = 0
        self.load_data(self.halo_id)
        
        
        
    def load_data(self, id):
        self.get_subhalo(id = id)
        self.get_particle_data(id = id)
        
    
    def get(self, path, params = None, name = None):
        os.makedirs(DATAPATH,exist_ok=True)
        r = requests.get(path, params=params, headers=self.headers)
        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()
        if r.headers['content-type'] == 'application/json':
            return r.json() # parse json responses automatically
        if 'content-disposition' in r.headers:
            filename = r.headers['content-disposition'].split("filename=")[1] if name is None else name
            with open(f"{DATAPATH}/{filename}_{self.halo_id}.hdf5", 'wb') as f:
                f.write(r.content)
            return filename # return the filename string

        return r
        
          
    def get_subhalo(self, id = None):
        if id is None: id = self.halo_id
        self.subhalo = self.get(f'{baseURL}/{self.simulation}/snapshots/{self.snapshot}/subhalos/{id}')
        self.center = np.array([self.subhalo["pos_x"],self.subhalo["pos_y"],self.subhalo["pos_z"]])
        self.mass = self.subhalo["mass"]
        self.halfmassrad = self.subhalo["halfmassrad_stars"]
    
    
    
    
       
    def get_particle_data(self,id = None):
        if id is None:id = self.halo_id
        print("loading data.")
        self.get(f'{baseURL}/{self.simulation}/snapshots/{self.snapshot}/subhalos/{id}/cutout.hdf5?stars=Coordinates,Masses,StellarHsml,GFM_StellarFormationTime', name = "stars")
        
        
        
        #self.get(f'{baseURL}/{self.simulation}/snapshots/{self.snapshot}/subhalos/{id}/cutout.hdf5?gas=Coordinates,Masses,StarFormationRate', name = "gas")
        with h5py.File(f"{DATAPATH}/stars_{self.halo_id}.hdf5", "r") as f:
            f = f["PartType4"]
            self.stars = dict(Coordinates = f["Coordinates"][()],GFM_StellarFormationTime=f["GFM_StellarFormationTime"][()],
                              Masses =f["Masses"][()],StellarHsml =f["StellarHsml"][()])
        os.remove(f"{DATAPATH}/stars_{self.halo_id}.hdf5")
        '''
        SOME GALAXIES HAVE TOO MUCH GAS PARTICLES, PROCESS GETS KILLED; CANT LOAD GALAXY
        with h5py.File(f"{DATAPATH}/gas_{self.halo_id}.hdf5", "r") as f:
            try:
                f = f["PartType0"]
                self.gas = dict(Coordinates = f["Coordinates"][()],StarFormationRate=f["StarFormationRate"][()],
                                Masses =f["Masses"][()])
            except:
                print("No Gas data.")
                self.gas_flag = False
        os.remove(f"{DATAPATH}/gas_{self.halo_id}.hdf5")
        '''  
        print("finished.")
        
    '''   
    def mass_querry(self,M_min=10,M_max = 12):
        # h = 0.704
        mass_min=10**M_min/1e10 *0.704
        mass_max = 10**M_max/1e10 * 0.704
        
        print(mass_min, mass_max)
        search_querry = "?mass__gt=" + str(mass_min) + "&mass__lt=" + str(mass_max)
        self.subhalo_querry = self.get(f"{baseURL}/{self.simulation}/snapshots/{self.snapshot}/subhalos/{search_querry}")
        
        self.ids_querry = []
        while self.subhalo_querry["next"] is not None: #???
            self.subhalo_querry = self.get(self.subhalo_querry["next"])
            self.ids_querry.append([[self.subhalo_querry['results'][i]['id'] for i in range(len(self.subhalo_querry["results"]))]])
        
        return(self.ids_querry)
    ''' 
        
    def face_on(self):
        # Das hier kann man später verallgemeinern
        if self.rotated_flag: return self.particles
        
        self.rotated_flag = 1
        inertie_tensor = viz.momentOfIntertia(stars = self.stars, gas = self.gas, rHalf=self.halfmassrad, subhalo_pos= self.center, gas_flag = self.gas_flag)
        rotation_matrix = viz.rotationMatrix(inertiaTenor=inertie_tensor, return_value = "face-on")
        
        pos = self.stars["Coordinates"]- self.center
        rot_pos =np.dot(rotation_matrix, pos.transpose()).transpose()
        self.particles = np.asarray(rot_pos)
        return(self.particles)

    
    
    def image2D(self, weights = "Masses", plot_factor = 1.5, res = 128, normed = False, return_hist = False):
        if not self.rotated_flag: self.face_on()
        img = viz.image2D(coordinates = self.particles, R_half = self.halfmassrad, 
                            weights=self.stars[weights], smoothing_length=self.stars["StellarHsml"], normed = normed,plot_factor=plot_factor, res =res)
        
        plt.imshow(img)
        if return_hist: return(img)
    
    def image3D(self, weights = "Masses", plot_factor = 1.5, res = 64, normed = False, return_hist = False,
                min_val = 1e-5, marker_size = 25, colorscale = "viridis", opacity = 0.6, log = True, show_plot = True):
        if not self.rotated_flag: self.face_on()
        img = viz.image3D(coordinates = self.particles, R_half = self.halfmassrad, 
                            weights=self.stars[weights], smoothing_length=self.stars["StellarHsml"], normed = normed,plot_factor=plot_factor, 
                                res =res,min_val = min_val, marker_size = marker_size, colorscale = colorscale, opacity = opacity, log = log, show_plot=show_plot)
        
        
        if return_hist: return(img)