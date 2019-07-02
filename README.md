# Recurrent-Models-of-Visual-Attention-TF-2.0
This repository contains the a modified Recurrent Attention Model which was described in the Paper Recurrent Models of Visual Attention. 
In order to run this code is it **recommended** to use the docker container of tensorflow 2.0.0a. Because `tensorflow-probability` doesn't support (from 17.04.19) tensorflow 2.0.0a, we need tf-nightly build. 

```bash
pip install --upgrade tf-nightly-gpu-2.0-preview tfp-nightly
```

## Requirements
- [ray](http://ray.readthedocs.io)
- [tensorflow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf)
- numpy, matplotlib, scikit-learn

## Modifications
- instead of translating and adding clutter while runtime, data loaders were created where this process is done only once. 
  - has the disadvantage/advantage that the amount of different images is limited
  - it is possible to see that the RAM is possible to archive a good performance with limited data
  - but you can create the dataset after each epoch to simulate the creation via runtime
- instead of Dense layers/Fully Conneted layers, Convolution layers were used
- in addition to the baseline model, batch norm was added to reduce variance
- instead of SGD with Momentum and learning rate decay, a ADAM optimizer was used
- instead of random search, Bayessian Hyperparameter Optimization was used to tune the hyperparameter of the network (std and initial learning rate)

## Project Structure
- `data/` contains scripts for loading data e.g. bach dataset loader, mnist, ...
- `model/` contains implementation of the whole model
    - `ram.py` contains the implementation of the Recurrent Attention Model
    - `layers.py` contains the implementation of the convolution layer (change this to try out other convolutions)
- `visualizations/` contain scripts for visualizing the model and data
- `./` contains jupyter notebooks about how to use the dataloader, how to use the visualization scripts and how to train the model

## Results
Coming soon

## Some Words
The paper Recurrent Models of Visual Attention is 5 years and received since then a lot of modification. I think the REINFORCE algorithm still a interesting "cheat" or "trick" to optimize for non differentiable variables which is why I tried to implement and understand it. This implementation also has a very object oriented style, thus every class/module can be swapped out easily.
