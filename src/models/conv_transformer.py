import torch
from torch import nn
import torch.nn.functional as F
from .resnet1d import BasicBlock1d


# Self Attention Class
class SelfAttentionConv(nn.Module):
    def __init__(self, k, headers=8, kernel_size=5, mask_next=True, mask_diag=False, stride=1):
        super().__init__()

        self.k, self.headers, self.kernel_size = k, headers, kernel_size
        self.mask_next = mask_next
        self.mask_diag = mask_diag

        h = headers

        # Query, Key and Value Transformations
        padding = kernel_size - stride
        self.padding_opertor = nn.ConstantPad1d((padding, 0), 0)
        self.toqueries = nn.Conv1d(k, k * h, kernel_size, stride=stride, padding=0, bias=True)
        self.tokeys = nn.Conv1d(k, k * h, kernel_size, stride=stride, padding=0, bias=True)
        self.tovalues = nn.Conv1d(
            k, k * h, kernel_size=kernel_size, stride=stride, padding=0, bias=False
        )  # No convolution operated

        # Heads unifier
        self.unifyheads = nn.Linear(k * h, k)
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        # Extraction dimensions
        b, t, k = x.size()  # batch_size, number_of_timesteps, number_of_time_series

        # Checking Embedding dimension
        assert self.k == k, (
            "Number of time series "
            + str(k)
            + " didn t much the number of k "
            + str(self.k)
            + " in the initiaalization of the attention layer."
        )
        h = self.headers

        #  Transpose to see the different time series as different channels
        x = x.transpose(1, 2)
        x_padded = self.padding_opertor(x)

        new_t = int(((t - self.kernel_size + self.padding) / self.stride) + 1)

        # Query, Key and Value Transformations
        queries = self.toqueries(x_padded).view(b, k, h, new_t)
        keys = self.tokeys(x_padded).view(b, k, h, new_t)
        values = self.tovalues(x_padded).view(b, k, h, new_t)

        # Transposition to return the canonical format
        queries = queries.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        queries = queries.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        values = values.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        values = values.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        keys = keys.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        keys = keys.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        # Weights
        queries = queries / (k ** (0.25))
        keys = keys / (k ** (0.25))

        queries = queries.transpose(1, 2).contiguous().view(b * h, new_t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, new_t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, new_t, k)

        weights = torch.bmm(queries, keys.transpose(1, 2))

        ## Mask the upper & diag of the attention matrix
        if self.mask_next:
            if self.mask_diag:
                indices = torch.triu_indices(new_t, new_t, offset=0)
                weights[:, indices[0], indices[1]] = float("-inf")
            else:
                indices = torch.triu_indices(new_t, new_t, offset=1)
                weights[:, indices[0], indices[1]] = float("-inf")

        # Softmax
        weights = F.softmax(weights, dim=2)

        # Output
        output = torch.bmm(weights, values)
        output = output.view(b, h, new_t, k)
        output = output.transpose(1, 2).contiguous().view(b, new_t, k * h)

        return self.unifyheads(output)  # shape (b,t,k)


# Conv Transforme Block


class ConvTransformerBLock(nn.Module):
    def __init__(
        self,
        k,
        headers,
        kernel_size=5,
        mask_next=True,
        mask_diag=False,
        dropout_proba=0.2,
        stride=1,
        multiplier_coefficient=32,
    ):
        super().__init__()

        # Self attention
        self.attention = SelfAttentionConv(k, headers, kernel_size, mask_next, mask_diag, stride)

        self.conv_reducer = nn.MaxPool1d(kernel_size, stride=stride)

        self.conv_reducer = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=kernel_size, stride=stride)

        # First & Second Norm
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        # Feed Forward Network
        self.feedforward = nn.Sequential(
            nn.Linear(k, multiplier_coefficient * k), nn.ELU(), nn.Linear(multiplier_coefficient * k, k)
        )
        # Dropout funtcion  & Relu:
        self.dropout = nn.Dropout(p=dropout_proba)
        self.activation = nn.ELU()

        padding = kernel_size - stride
        self.padding_opertor = nn.ConstantPad1d((padding, 0), 0)

    def forward(self, x, train=True):
        # Self attention + Residual
        x_attention = self.attention(x)
        x_reduced = self.conv_reducer(self.padding_opertor(x.transpose(1, 2))).transpose(1, 2)

        # Dropout attention
        x = x_attention + x_reduced
        if train:
            x = self.dropout(x)

        # First Normalization
        x = self.norm1(x)

        # Feed Froward network + residual
        x = self.feedforward(x) + x

        # Second Normalization
        x = self.norm2(x)

        return x


# Forcasting Conv Transformer :
class ForcastConvTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        k,
        headers,
        depth,
        seq_length,
        kernel_size=5,
        mask_next=True,
        mask_diag=False,
        dropout_proba=0.2,
        num_tokens=None,
        dd1=0.5,
        stride=1,
        multiplier_coefficient=32,
        **kwargs,
    ):
        super().__init__()
        # Embedding
        self.tokens_in_count = False

        self.encoder = BasicBlock1d()

        if num_tokens:
            self.tokens_in_count = True
            self.token_embedding = nn.Embedding(num_tokens, k)

        # Embedding the position
        self.position_embedding = nn.Embedding(seq_length, k)

        # Number of time series
        self.k = k
        self.seq_length = seq_length

        # Transformer blocks
        tblocks = []
        for _ in range(depth):
            block = ConvTransformerBLock(
                k,
                headers,
                kernel_size,
                mask_next,
                mask_diag,
                dropout_proba,
                stride,
                multiplier_coefficient=multiplier_coefficient,
            )
            tblocks.append(block)
        self.TransformerBlocks = nn.Sequential(*tblocks)

        # Transformation from k dimension to numClasses
        self.flatten = nn.Flatten()
        # self.topreSigma = nn.Linear(k, num_classes)
        result_seq_len = self.seq_length
        for _ in range(depth):
            result_seq_len = int(((result_seq_len - kernel_size + (kernel_size - stride)) / stride) + 1)

        self.lstm = nn.LSTM(self.k, self.k * 2, batch_first=True)
        self.tomu = nn.Linear(result_seq_len * self.k, num_classes)
        self.plus = nn.Softplus()
        self.dd1 = nn.Dropout(p=dd1)

    def forward(self, x, tokens=None):
        x = self.encoder(x)

        x = x.permute(0, 2, 1)
        b, t, k = x.size()

        # checking that the given batch had same number of time series as the BLock had
        assert k == self.k, (
            "The k :"
            + str(self.k)
            + " number of timeseries given in the initialization is different than what given in the x :"
            + str(k)
        )
        assert t == self.seq_length, (
            "The lenght of the timeseries given t "
            + str(t)
            + " miss much with the lenght sequence given in the Tranformers initialisation self.seq_length: "
            + str(self.seq_length)
        )

        # Position embedding
        pos = torch.arange(t).to(device=x.device)
        self.pos_emb = self.position_embedding(pos).expand(b, t, k)

        # Checking token embedding
        assert self.tokens_in_count == (tokens is not None), (
            "self.tokens_in_count = "
            + str(self.tokens_in_count)
            + " should be equal to (not (tokens is None)) = "
            + str((tokens is not None))
        )
        if tokens is not None:
            ## checking that the number of tockens corresponde to the number of batch elements
            assert tokens.size(0) == b
            self.tok_emb = self.token_embedding(tokens)
            self.tok_emb = self.tok_emb.expand(t, b, k).transpose(0, 1)

        # Adding Pos Embedding and token Embedding to the variable
        if tokens is not None:
            x = self.pos_emb + self.tok_emb + x
        else:
            x = self.pos_emb + x

        # Transformer :
        for block in self.TransformerBlocks:
            x = block(x)

        # x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.dd1(x)
        x = self.tomu(x)
        # sigma = self.plus(presigma)
        # x = self.plus(x)

        return x


def conv_transformer(**kwargs):
    return ForcastConvTransformer(**kwargs)
