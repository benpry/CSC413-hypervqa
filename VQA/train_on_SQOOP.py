"""
This file reads the SQOOP data and trains a simple model to answer questions about it
"""
import numpy as np
import h5py
import vr
from vr import utils
from vr.data import ClevrDataLoader

DATA_DIR = "../data/sqoop-variety_8-repeats_3750"
BS = 64

if __name__ == "__main__":

    vocab = utils.load_vocab(f"{DATA_DIR}/vocab.json")

    train_loader_kwargs = {
      'question_h5': f"{DATA_DIR}/train_questions.h5",
      'feature_h5': f"{DATA_DIR}/train_features.h5",
      'load_features': 0,
      'vocab': vocab,
      'batch_size': BS,
      'shuffle': True,
      'question_families': None,
      'max_samples': None,
      'num_workers': 0,
      'percent_of_data': 1,
    }

    train_loader = ClevrDataLoader(**train_loader_kwargs)

    for batch in train_loader:
        questions, _, feats, answers, programs, _ = batch
        print(feats.shape)
