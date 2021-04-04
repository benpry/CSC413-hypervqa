"""
This file contains the specification of the model
"""
import torch
from torch import nn
from torch.nn import functional as F


class HyperVQA(nn.Module):

    def __init__(self, vocab_size, embedding_dim, encoder_hidden_size, conv_shapes=()):
        """Initialize the modules"""
        super(HyperVQA, self).__init__()
        self.conv_shapes = conv_shapes

        total_output_weights = 0
        prev_I = 3
        for layer in conv_shapes:
            I, K = layer
            total_output_weights += I * prev_I * (K ** 2)
            prev_I = I

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, encoder_hidden_size)
        self.hypernet = nn.Sequential(
            nn.Linear(encoder_hidden_size, encoder_hidden_size * 4),
            nn.ReLU(),
            nn.Linear(encoder_hidden_size * 4, encoder_hidden_size * 8),
            nn.ReLU(),
            nn.Linear(encoder_hidden_size * 8, total_output_weights)
        )

        self.linear = nn.Sequential(
            nn.Linear(conv_shapes[-1][0] * (64 ** 2), 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def conv_batch(self, x, weights, biases):
        """Apply the given weights and biases to the elements, one per batch"""
        n_items = x.shape[0]
        item_outputs = []
        # TODO: figure out how to vectorize this
        for i in range(n_items):
            item_weights = weights[i, ...]
            item_bias = biases[i, ...]
            item = x[i, ...].unsqueeze(0)
            item_out = F.conv2d(item, item_weights, item_bias, padding=1)
            item_outputs.append(item_out)

        all_output = torch.cat(item_outputs, dim=0)
        return all_output

    def forward(self, questions, images):
        """Make a prediction based on questions and images"""
        word_embeddings = self.embedding(questions).transpose(1, 0)

        # encode into task_codes by taking the last hidden state with an LSTM
        task_codes = self.encoder(word_embeddings)[1][0].squeeze()

        # get the weights from the hypernetwork
        all_conv_weights = self.hypernet(task_codes)

        # reshape the weights for the convolutional neural net
        batch_size = all_conv_weights.shape[0]

        x = images
        prev_I = 3
        # apply the convolutional layers
        for conv_layer in self.conv_shapes:
            I, K = conv_layer
            layer_num_weights = prev_I * I * (K ** 2)
            conv_weights, conv_biases = all_conv_weights[..., :layer_num_weights], all_conv_weights[..., layer_num_weights:layer_num_weights+I]
            conv_weights = conv_weights.view(batch_size, I, prev_I, K, K)
            x = self.conv_batch(x, conv_weights, conv_biases)
            prev_I = I

        # apply the final linear layers
        x = x.flatten(1)
        x = self.linear(x)

        # return logits
        return x
