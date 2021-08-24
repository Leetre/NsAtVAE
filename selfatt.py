import torch, math
import torch.nn as nn

class SelfAttention(nn.Module):
    "self attention model"
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # Parameter definition
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)

        # query, key, value
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_embeddings):
        mixed_query_layer = self.query(input_embeddings)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(input_embeddings)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(input_embeddings)  # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

if __name__ == '__main__':
    attention = SelfAttention(4, 2, 0.2)
    x_in = torch.randn(1, 3, 4)
    x_out = attention(x_in)
    print(x_out.shape, '\n', x_out)
