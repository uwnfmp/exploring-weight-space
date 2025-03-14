# Exploring the Weight Space of Neural Networks through Classification and Generation

This is the code repository of the paper "Exploring the Weight Space of Neural Networks for Learning and Generation". There are two main objectives in this project. First is to explore the weight space of the classifier MLP zoos and train other models to classify the features of the dataset used to train these zoos. The model architectures used to train parameter classifiers are: MLP, Set Transformers, DWSNets. Second is to train a generational models in the weight space of these zoos to conditionaly generate new set of parameters. We compare the generated parameters between Variational Autoencoder, Diffusion and Condition Flow Matching models.

## Structure

- src - contains the reusable code used throughout the research.
- data - should contain dataset files for each zoo and data splits. Because of the big sizes they can be downloaded from another [repository](https://github.com/uwnfmp/exploring-weight-space-data).
- models - should contains saved models. Can be downloaded same as <b>data</b>.
- configs - contains the .yaml config files for the MLP classifiers.
- reports - contains plots and diagrams with results.
- notebooks - contains jupyter notebooks used to train and evaluate classifier and generative models.

## Interesting notebooks

- dataset.ipynb - contains code used to train the model zoos and their conversion into a dataset format.
- classification.ipynb - contains the evaluation of classifier models.
- generation.ipynb - contains the evaluation of generational models.
- models folder - contains notebooks with training code for all the models (except model zoos, look dataset.ipynb) used in this project.
