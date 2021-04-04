"""
This file reads the SQOOP data and trains a simple model to answer questions about it
"""
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from vr import utils
from vr.data import ClevrDataLoader
from model import HyperVQA

DATA_DIR = "../data/sqoop-variety_8-repeats_3750"
BS = 256

device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = HyperVQA(50, 64, 32, conv_shapes=[(8, 3), (16, 3), (32, 3)])

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    for batch in train_loader:
        questions, _, feats, answers, programs, _ = batch

        optimizer.zero_grad()
        pred = model(questions, feats)
        loss = loss_fn(pred, answers)
        print(f"loss: {loss.item()}")
        loss.backward()
        optimizer.step()
