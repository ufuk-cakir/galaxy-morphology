# Machine Learning Models For Galaxy Morphology

This is the GitHub repository for my bachelor thesis in physics at the Heidelberg University.

# Introduction
State-of-the-art hydrodynamical simulations can reproduce the diversity of different galaxy morphological types, but fail to exactly recreate real, observed galaxies.
In the last decade, machine learning (ML) had very promising results in image recognition and dimensionality reduction. As an example of what ML is able to achieve, see the mind-blowing results of DALL·E 2, which can generate images from natural language. And since ML has proven to be a powerful tool, why not use it for astrophysical purposes? That is why the goal of this thesis is to investigate how ML can be used to create galaxy morphology models and encode the information contained in modern state-of-the-art simulations.

## Principal Component Analysis
The main part of my thesis is to investigate of how principal component analysis(PCA) can serve as a galaxy morphology model. The so called "Eigengalaxies" calculated from PCA are galaxy images which act as the basis vectors of the image space such that each galaxy in the dataset can be described through a linear combination of all eigengalaxies.

Reconstruction of a Galaxy:
![screen-gif](./animations/6028.gif)
![screen-gif](./animations/5237.gif)


## Generative Advesarial Neural Network
Traininieren des [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada) 



