#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from vr.models.baselines import LstmEncoder
from vr.models.baselines import build_mlp


class HyperVQA(nn.Module):
    """A model that uses a HyperNetwork to produce the weights for a CNN"""
    def __init__(self, vocab, conv_layers=(8, 16, 32), conv_kernels=(3, 3, 3), rnn_wordvec_dim=128, rnn_dim=256,
                 rnn_num_layers=2, rnn_dropout=0, fc_dims=(1024,), fc_use_batchnorm=False, fc_dropout=0):
        super().__init__()

        assert len(conv_layers) == len(conv_kernels)

        total_output_weights = 0
        prev_I = 3
        self.conv_shapes = []
        for i in range(len(conv_layers)):
            I = conv_layers[i]
            K = conv_kernels[i]
            self.conv_shapes.append((I, K))
            total_output_weights += I * prev_I * (K ** 2) + I
            prev_I = I

        rnn_kwargs = {
            'token_to_idx': vocab['question_token_to_idx'],
            'wordvec_dim': rnn_wordvec_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': rnn_dropout
        }
        self.rnn = LstmEncoder(**rnn_kwargs)

        self.hypernet = nn.Sequential(
            nn.Linear(rnn_dim, rnn_dim * 4),
            nn.LeakyReLU(negative_slope=1/5.5),
            nn.Linear(rnn_dim * 4, rnn_dim * 16),
            nn.LeakyReLU(negative_slope=1/5.5),
            nn.Linear(rnn_dim * 16, total_output_weights)
        )

        classifier_kwargs = {
            'input_dim': 2048,  # self.conv_shapes[-1][0] * (64 // (2 ** (len(conv_layers)+1))) + rnn_dim,
            'hidden_dims': fc_dims,
            'output_dim': len(vocab['answer_token_to_idx']),
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
        }

        self.linear = build_mlp(**classifier_kwargs)

    def conv_batch(self, x, weights, biases):
        """Apply the given weights and biases to the elements, one per batch"""

        # Main idea: combining batch dimension and channel dimension
        n_items = x.shape[0]
        original_out_channels = weights.shape[1]
        out_channels = n_items*original_out_channels
        
        all_output = F.conv2d(x.view(1, n_items*x.shape[1], x.shape[2], x.shape[3]), weights.reshape(out_channels, weights.shape[2], weights.shape[3], weights.shape[4]), biases.reshape(-1), padding=1, groups=n_items)
        
        return all_output.reshape(n_items, original_out_channels, all_output.shape[2], all_output.shape[3])

    def forward(self, questions, feats):
        """Make a pass with the HyperVQA model"""
        # encode the questions and get corresponding weights
        q_feats = self.rnn(questions)
        all_conv_weights = self.hypernet(q_feats)

        batch_size = all_conv_weights.shape[0]
        x = feats
        prev_I = 3
        # apply convolutional layers
        for conv_layer in self.conv_shapes:
            I, K = conv_layer
            layer_num_weights = prev_I * I * (K ** 2)
            conv_weights, conv_biases = all_conv_weights[..., :layer_num_weights],\
                all_conv_weights[..., layer_num_weights:layer_num_weights + I]

            conv_weights = conv_weights.view(batch_size, I, prev_I, K, K)
            x = self.conv_batch(x, conv_weights, conv_biases)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            prev_I = I
            all_conv_weights = all_conv_weights[..., layer_num_weights + I:]

        x = x.flatten(1)
        x = self.linear(x)

        # return logits
        return x
