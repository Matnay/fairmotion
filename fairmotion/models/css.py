# Author: Pranay Mathur

import random
import torch
import torch.nn as nn
from .css_encoder_decoder import cssEncoder, cssDecoder, cssEncoderShort


class convSeq2Seq(nn.Module):
    """convSeq2Seq model for sequence prediction. The model uses a single convSeq2Seq module to
    take a sliding window of input poses, as well as the entire sequence as an embedding and generates 
    a pose prediction for the next time step.

    Attributes:
        input_dim: Size of input vector for each time step
        hidden_dim: convSeq2Seq hidden size
        dropout: Probability of an element to be zeroed
        device: Device on which to run the convSeq2Seq module
    """

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1, device="cuda"):
        super(convSeq2Seq, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.device = device

        self.sliding_window_size = 2

        self.long_term_encoder = cssEncoder(
            input_dim, hidden_dim).to(device=device)
        self.long_term_encoder = self.long_term_encoder.double()

        self.short_term_encoder = cssEncoderShort(
            input_dim, hidden_dim).to(device=device)
        self.short_term_encoder = self.short_term_encoder.double()

        self.short_term_decoder = cssDecoder(
            input_dim, hidden_dim).to(device=device)
        self.short_term_decoder = self.short_term_decoder.double()

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def run_short_term_enc_dec(self, inputs, outputs, max_len=None, z_l_embedding=None, teacher_forcing_ratio=0):
        if max_len is None:
            max_len = inputs.shape[2]
        else:
            teacher_forcing_ratio = 0

        for t in range(1, max_len):
            if t >= inputs.shape[2]:
                input = output.unsqueeze(1)
            else:
                input = inputs[:, :, t].unsqueeze(1)

            teacher_force = random.random() < teacher_forcing_ratio
            if t < self.sliding_window_size:
                input = inputs[:, -1].unsqueeze(1).unsqueeze(1)
            else:
                input = outputs[:, t -
                                self.sliding_window_size].unsqueeze(1).unsqueeze(1)

            input = input.double()

            prev_output = outputs[:, t-1].unsqueeze(1).unsqueeze(1)

            short_term_input = torch.cat((input, prev_output), dim=2)
            
            # run short term encoder to generate output
            output = self.short_term_encoder(short_term_input)

            # concatenate inputs and long term embedding
            z_sl_embedding = torch.cat((output, z_l_embedding), dim=-1)
            output = self.short_term_decoder(z_sl_embedding)
            outputs[:, t] = output

        return outputs

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        """
        Inputs:
            src, tgt: Tensors of shape (batch_size, seq_len, input_dim)
            max_len: Maximum length of sequence to be generated during
                inference. Set None during training.
            teacher_forcing_ratio: Probability of feeding gold target pose as
                decoder input instead of predicted pose from previous time step
        """
        model_input = self.dropout(src)

        # Generate as many poses as in tgt during training
        max_len = tgt.shape[1] if max_len is None else max_len

        model_input = model_input.unsqueeze(1)
        z_le = self.long_term_encoder(model_input)

        if self.training:
            outputs = torch.zeros(
                src.shape[0], max_len, src.shape[2]).to(self.device)
            outputs[:, 0] = src[:, -1]
            outputs = self.run_short_term_enc_dec(
                src, outputs, max_len=max_len, z_l_embedding=z_le, teacher_forcing_ratio=teacher_forcing_ratio)
        else:
            outputs = torch.zeros(
                src.shape[0], max_len, src.shape[2]).to(self.device)
            inputs = src
            outputs = self.run_short_term_enc_dec(
                inputs, outputs, max_len=max_len, z_l_embedding=z_le, teacher_forcing_ratio=0)
        return outputs
