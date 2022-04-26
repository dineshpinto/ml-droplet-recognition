# Droplet Detection with Neural Networks

Use a Convolutional Neural Network built in TensorFlow and Keras to detect a droplet in an experimental data set.

The neural net consists of 4 layers, and for testing data shows reasonable results as shown below:

![neural_net_results](results/test_data_result.png)

The goal is to apply it to a biological sample and detect droplet formation. The droplet as imaged by a microscope 
looks like:
<img src="results/real_data_raw.png" width="250" height="200">

A thresholding algorithm is applied to the raw image and the droplet is labelled:
![processed_droplet](results/real_data_result.jpg)

This is performed for a large enough set of microscope images, and used to train the neural network. 
The resulting trained model is then used on real data.

**Note:** All biological droplet data sourced from [@cfsb618](https://github.com/cfsb618)

## Model
![keras_model](results/model.png)

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
