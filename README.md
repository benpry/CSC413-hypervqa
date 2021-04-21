# Visual Question-Answering with HyperNetworks

This repository contains code for the CSC413 project "Visual Question-Answering with HyperNetworks" by Ben Prystawski and Utkarsh Agarwal. 

The main file of interest is vr/models/hypervqa.py, which contains our implementation of the HyperVQA model. We also modified scripts/train_model.py to make it compatible with HyperVQA and scripts/train/hypervqa_flatqa.sh to train HyperVQA with the hyperparameters and architecture we specify. The notebook TrainHyperVQA.ipynb can be used to train this model on Google Colab. CreateVisualizations.ipynb simply visualizes a few images from the SQOOP dataset. 

## Acknowledgement

Code for this project is based on the [repository](https://github.com/rizar/systematic-generalization-sqoop) for "[Systematic Generalization: What is Required and Can It Be Learned?](https://arxiv.org/abs/1811.12889)" by Dzmitry Bahdanau, Shikhar Murty, Michael Noukhovitch, Thien Huu Nguyen, Harm de Vries, and Aaron Courville. Their code is in turn based on the [code](https://github.com/ethanjperez/film) for [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871v2) by Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. Their code is based on the [code](https://github.com/facebookresearch/clevr-iep) for "[Inferring and Executing Programs for Visual Reasoning](https://arxiv.org/abs/1705.03633)." by Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Judy Hoffmman, Fei-Fei Li, Larry Zitnick, and Ross Girshick.

We stand atop a long chain of giants standing upon the shoulders of other giants.
