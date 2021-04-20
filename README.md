# Visual Question-Answering with HyperNetworks

This repository contains code for the CSC413 project "Visual Question-Answering with HyperNetworks" by Ben Prystawski and Utkarsh Agarwal. Code is heavily based on Dzmitry Bahdanau et al.'s [repository](https://github.com/rizar/systematic-generalization-sqoop) for "[Systematic Generalization: What is Required and Can It Be Learned?](https://arxiv.org/abs/1811.12889)" which is in turn based on the [code](https://github.com/ethanjperez/film) for Perez et al.'s [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871v2).

The main file we worked in is vr/models/hypervqa.py, which contains our implementation of the HyperVQA model. We also modified scripts/train_model.py to make it compatible with HyperVQA and scripts/train/hypervqa_flatqa.sh to train HyperVQA with the hyperparameters and architecture we specify. Finally, the notebook TrainHyperVQA.ipynb can be used to train this model on Google Colab. CreateVisualizations.ipynb simply visualizes a few images from the SQOOP dataset. 