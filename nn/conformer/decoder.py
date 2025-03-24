# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from torch import Tensor
from typing import Tuple
import torch.nn.init as init


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class DecoderRNNT(nn.Module):
    """
    Decoder of RNN-Transducer

        Args:
        input_size (int, optional): Input feature dimension (default: 136)
        hidden_size (int, optional): Hidden state dimension of decoder (default: 512)
        output_dim (int, optional): Output dimension of the decoder (default: 128)
        num_layers (int, optional): Number of decoder layers (default: 1)
        rnn_type (str, optional): Type of RNN cell (default: 'lstm')
        dropout_p (float, optional): Dropout probability of the decoder (default: 0.2)
    """

    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            input_size: int = 136,
            hidden_size: int = 512,
            output_dim: int = 128,
            num_layers: int = 1,
            rnn_type: str = 'lstm',
            dropout_p: float = 0.2,
    ):
        super(DecoderRNNT, self).__init__()
        self.hidden_size = hidden_size

        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )
        self.out_proj = Linear(hidden_size, output_dim)

    def count_parameters(self) -> int:
        """ Count parameters of decoder """
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of decoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor = None,
            hidden_states: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` (targets) for training.

        Args:
            inputs (torch.FloatTensor): A sequence of previous acoustic features. FloatTensor of shape (batch, seq_length, input_size)
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            hidden_states (torch.FloatTensor): A previous hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``

        Returns:
            (Tensor, Tensor):

            * decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * hidden_states (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        """
        rnn_inputs = inputs

        if input_lengths is not None:
            rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, input_lengths.cpu(), batch_first=True)
            outputs, hidden_states = self.rnn(rnn_inputs, hidden_states)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = self.out_proj(outputs)
        else:
            outputs, hidden_states = self.rnn(rnn_inputs, hidden_states)
            outputs = self.out_proj(outputs)

        return outputs, hidden_states
