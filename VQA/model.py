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
            total_output_weights += I * prev_I * (K ** 2) + I
            prev_I = I

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, encoder_hidden_size)
        self.hypernet = nn.Sequential(
            nn.Linear(encoder_hidden_size, encoder_hidden_size * 4),
            nn.LeakyReLU(negative_slope=1/5.5),
            nn.Linear(encoder_hidden_size * 4, encoder_hidden_size * 16),
            nn.LeakyReLU(negative_slope=1/5.5),
            nn.Linear(encoder_hidden_size * 16, total_output_weights)
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

        # Main idea: combining batch dimension and channel dimension
        n_items = x.shape[0]
        original_out_channels = weights.shape[1]
        out_channels = n_items*original_out_channels
        
        all_output = F.conv2d(x.view(1, n_items*x.shape[1], x.shape[2], x.shape[3]), weights.reshape(out_channels, weights.shape[2], weights.shape[3], weights.shape[4]), biases.reshape(-1), padding=1, groups=n_items)
        
        return all_output.reshape(n_items, original_out_channels, all_output.shape[2], all_output.shape[3])

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
            all_conv_weights = all_conv_weights[..., layer_num_weights+I:]

        # apply the final linear layers
        x = x.flatten(1)
        x = self.linear(x)

        # return logits
        return x
