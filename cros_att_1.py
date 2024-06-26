import torch
from torch import nn
import math


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class CrossAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(CrossAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        attention_probs_dropout_prob = 0.2
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        # 
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, y):
        # x and y are the input tensors
        # Assume x has shape [batch_size, seq_len_x, input_size]
        # Assume y has shape [batch_size, seq_len_y, input_size]
        # mixed_query_layer = self.query(torch.cat([x, y], dim=1))

        mixed_query_layer = self.query(x)
        # mixed_query_layer has shape [batch_size, seq_len_x, hidden_size]
        # The query linear layer is applied to x

        mixed_key_layer = self.key(y)
        # mixed_key_layer has shape [batch_size, seq_len_y, hidden_size]
        # The key linear layer is applied to y

        mixed_value_layer = self.value(y)
        # mixed_value_layer has shape [batch_size, seq_len_y, hidden_size]
        # The value linear layer is applied to y

        query_layer = self.transpose_for_scores(mixed_query_layer)
        # query_layer has shape [batch_size, num_attention_heads, seq_len_x, attention_head_size]
        # The mixed_query_layer is reshaped and transposed

        key_layer = self.transpose_for_scores(mixed_key_layer)
        # key_layer has shape [batch_size, num_attention_heads, seq_len_y, attention_head_size]
        # The mixed_key_layer is reshaped and transposed

        value_layer = self.transpose_for_scores(mixed_value_layer)
        # value_layer has shape [batch_size, num_attention_heads, seq_len_y, attention_head_size]
        # The mixed_value_layer is reshaped and transposed

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores has shape [batch_size, num_attention_heads, seq_len_x, seq_len_y]
        # The dot product is computed between query_layer and the transposed key_layer

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores has shape [batch_size, num_attention_heads, seq_len_x, seq_len_y]
        # The attention scores are scaled by the square root of attention_head_size

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs has shape [batch_size, num_attention_heads, seq_len_x, seq_len_y]
        # The attention scores are normalized using softmax

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        # attention_probs has shape [batch_size, num_attention_heads, seq_len_x, seq_len_y]
        # Dropout is applied to the attention probabilities

        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer has shape [batch_size, num_attention_heads, seq_len_x, attention_head_size]
        # The attention probabilities are multiplied with the value_layer

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # context_layer has shape [batch_size, seq_len_x, num_attention_heads, attention_head_size]
        # The dimensions are permuted

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer has shape [batch_size, seq_len_x, hidden_size]
        # The context_layer is reshaped to merge the attention heads

        hidden_states = self.dense(context_layer)
        # hidden_states has shape [batch_size, seq_len_x, hidden_size]
        # The dense linear layer is applied to the context_layer

        hidden_states = self.out_dropout(hidden_states)
        # hidden_states has shape [batch_size, seq_len_x, hidden_size]
        # Dropout is applied to the hidden_states

        # hidden_states = self.LayerNorm(hidden_states + torch.cat([x, y], dim=1))  # residual
        hidden_states = self.LayerNorm(hidden_states + x)  # residual
        # hidden_states has shape [batch_size, seq_len_x, hidden_size]
        # Layer normalization is applied to the sum of hidden_states and the input x

        return hidden_states
