__doc__ = """
This file contains the very vanilla implementation of the GPT2 model.

Or what we consider as baseline.

Let's make most of these parts reusable. So many of our experiment results will be reuseable.
"""

import math
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicOutput(SimpleNamespace):
    """
    Basic output of the model.
    """

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


class MLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        # 1st dense layer
        self.c_fc = nn.Linear(hidden_size, 4 * hidden_size)
        # 2nd dense layer
        self.c_proj = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        F.gelu(self.c_fc(x), approximate="tanh") is the activation function in between.
        It's not necessary the best, but can be easily compared to out of box implementations in pytorch / huggingface.
        """
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, n_head: int, n_positions: int):
        super().__init__()
        assert hidden_size % n_head == 0
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_head

        # c_attn is the linear projection for the query, key, and value
        # instead of using 3 separate linear layers, we can use a single linear layer and split the output.
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)

        # c_proj is the linear projection for the output.
        self.c_proj = nn.Linear(hidden_size, hidden_size)

        # standard decoder mask that will be multiplied to the attention matrix
        # so the future tokens are not leaked to the current token

        # obviously, we can save this mask accross the entire model, not every layer has its own decoder mask.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(n_positions, n_positions)).view(1, 1, n_positions, n_positions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        - B: batch size
        - T: sequence length
        - C: hidden size
        """
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)

        # the latent size for actual attention operation is hidden_size / n_head = head_dim.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        # apply the causal mask (sliced to the current context length) to the attention matrix
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        # the view(B, T, C) flattens head_dim x n_head back into the original hidden size.
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class TransformerBlock(nn.Module):
    """
    Transformer Block is wrapping the attention with:
    * MLP and
    * each through the residual connection.
    And goes through the layer normalization before each operation.
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        n_positions: int,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, n_head, n_positions)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BaselineGPT(nn.Module):
    """
    Baseline GPT model.
    """

    def __init__(
        self,
        vocab_size: int,
        n_positions: int,
        hidden_size: int,
        n_layer: int,
        n_head: int,
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.wpe = nn.Embedding(n_positions, hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size, n_head, n_positions) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)

        x = self.wte(input_ids) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        return BasicOutput(loss=loss, logits=logits)
