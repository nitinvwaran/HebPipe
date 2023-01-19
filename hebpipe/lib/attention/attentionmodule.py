"""
From Stanza
"""

import torch
import torch.nn as nn

class SoftDotAttention(nn.Module):
    """Soft Dot Attention.
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

        self.INFINITY_NUMBER = 1e12

    def forward(self, input, context, mask=None, attn_only=False, return_logattn=False):
        """Propagate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL

        if mask is not None:
            # sett the padding attention logits to -inf
            assert mask.size() == attn.size(), "Mask size must match the attention size!"
            attn.masked_fill_(mask, -self.INFINITY_NUMBER)

        if return_logattn:
            attn = torch.log_softmax(attn, 1)
            attn_w = torch.exp(attn)
        else:
            attn = self.sm(attn)
            attn_w = attn
        if attn_only:
            return attn

        attn3 = attn_w.view(attn_w.size(0), 1, attn_w.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class LSTMAttention(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True, attn_type='soft'):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        #if attn_type == 'soft':
        self.attention_layer = SoftDotAttention(hidden_size)
        """
        elif attn_type == 'mlp':
            self.attention_layer = BasicAttention(hidden_size)
        elif attn_type == 'linear':
            self.attention_layer = LinearAttention(hidden_size)
        elif attn_type == 'deep':
            self.attention_layer = DeepAttention(hidden_size)
        else:
            raise Exception("Unsupported LSTM attention type: {}".format(attn_type))
        """


    def forward(self, input, hidden, ctx, ctx_mask=None, return_logattn=False):
        """Propagate input through the network."""
        if self.batch_first:
            input = input.transpose(0,1)

        output = []
        attn = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.lstm_cell(input[i], hidden)
            hy, cy = hidden
            h_tilde, alpha = self.attention_layer(hy, ctx, mask=ctx_mask, return_logattn=return_logattn)
            output.append(h_tilde)
            attn.append(alpha)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0,1)

        if return_logattn:
            attn = torch.stack(attn, 0)
            if self.batch_first:
                attn = attn.transpose(0, 1)
            return output, hidden, attn

        return output, hidden