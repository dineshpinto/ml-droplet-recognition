# Droplet Detection with Neural Networks

Use a Convolutional Neural Network (CNN) built in TensorFlow and Keras to detect a droplet in an experimental data set.

## Installation
1. Create the conda environment from file (for Mac M1)
```shell
conda env create --file conda-env-macm1.yml
```
3. Activate environment 
```shell
conda activate ml_droplet
```
4. Add environment to Jupyter kernel 
```shell
python -m ipykernel install --name=ml_droplet
```
5. Explore the Jupyterlab Notebooks
```shell
jupyter lab
```


### Export conda environment
```shell
conda env export --no-builds | grep -v "^prefix: " > conda-env.yml
```
