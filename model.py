import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).to(device)
        embeddings = self.embeddings_table[final_mat].to(device)

        return embeddings


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 64

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([self.head_dim])))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        # [batch_size, q_len, n_heads, head_dim] -> [batch_size, n_heads, q_len, head_dim]
        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # [batch_size, k_len, n_heads, head_dim] -> [batch_size, n_heads, k_len, head_dim]
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # [batch_size, n_heads, q_len, k_len]
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        # [q_len, batch_size, hid_dim] -> [q_len, batch_size * n_heads, head_dim]
        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        # [q_len, k_len, head_dim]
        r_k2 = self.relative_position_k(len_q, len_k)
        # [q_len, batch_size * n_heads, head_dim] x [q_len, head_dim, k_len] ->
        # [q_len, batch_size * n_heads, k_len] -> [batch_size * n_heads, q_len, k_len]
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        # [batch_size * n_heads, q_len, k_len] -> [batch_size, n_heads, q_len, k_len]
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        # attn = [batch_size, n_heads, q_len, k_len]
        # [batch_size, k_len, n_heads, head_dim] -> [batch_size, n_heads, k_len, head_dim]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # [batch_size, n_heads, q_len, head_dim]
        weight1 = torch.matmul(attn, r_v1)

        # [q_len, k_len, head_dim]
        r_v2 = self.relative_position_v(len_q, len_v)
        # [q_len, batch_size * n_heads, k_len]
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        # [q_len, batch_size * n_heads, k_len] x [q_len, k_len, head_dim] -> [q_len, batch_size * n_heads, head_dim] 
        weight2 = torch.matmul(weight2, r_v2)
        # [batch_size, n_heads, q_len, head_dim] 
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        # [batch_size, q_len, n_heads, head_dim] 
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # [batch_size, q_len, hid_dim] 
        x = x.view(batch_size, -1, self.hid_dim)
        
        # x = [batch size, query len, hid dim]
        x = self.fc_o(x)
        
        # x = [batch size, query len, hid dim]
        return x


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pos_ff_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pos_ff_dim)
        self.fc_2 = nn.Linear(pos_ff_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        # x = [batch size, seq len, pf dim]
        x = self.fc_2(x)
        
        # x = [batch size, seq len, hid dim]
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pos_ff_dim, dropout):
        super(EncoderLayer, self).__init__()

        self.attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pos_ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # self-attention
        src_attn = self.self_attention(src, src, src, src_mask)
        src_attn = self.dropout(src_attn)

        # residual connection + layer norm
        src = self.attn_layer_norm(src + src_attn)
        
        # positionwise feedforward
        src_ff = self.positionwise_feedforward(src)
        src_ff = self.dropout(src_ff)

        # residual connection + layer norm
        src = self.ff_layer_norm(src + src_ff)

        return src


class GestureEncoder(nn.Module):
    def __init__(self, num_layers, n_heads, in_dim, hid_dim, out_dim, max_len, pos_ff_dim, dropout):
        super(GestureEncoder, self).__init__()

        reduce_dim = 128
        self.conv = nn.Conv1d(in_dim, hid_dim, kernel_size=5, padding=2)
        self.bn_1 = nn.BatchNorm1d(hid_dim, momentum=0.9)
        self.bn_2 = nn.BatchNorm1d(reduce_dim, momentum=0.9)
        
        self.encoder = nn.TransformerEncoder(
            EncoderLayer(hid_dim, n_heads, pos_ff_dim, dropout), 
            num_layers
        )

        self.fc = nn.Linear(hid_dim, reduce_dim)
        self.relu = nn.ReLU()

        self.fc_mean = nn.Linear(max_len * reduce_dim, out_dim)
        self.fc_logvar = nn.Linear(max_len * reduce_dim, out_dim)

    def forward(self, src): # [640, 69]
        batch_size = src.size()[0]
        # [batch_size, max_len, in_dim] -> [batch_size, in_dim, max_len]
        src = src.permute(0, 2, 1).contiguous()
        # [batch_size, hid_dim, max_len]
        src = self.bn_1(self.conv(src)).permute(0, 2, 1).contiguous()
        # [batch_size, max_len, hid_dim]
        src = self.encoder(src)
        # [batch_size, max_len, reduce_dim]
        src = self.fc(src).permute(0, 2, 1).contiguous()
        src = self.bn_2(src).permute(0, 2, 1).contiguous()
        src = self.relu(src)

        # [batch_size, max_len * reduce_dim]
        flatten_src = src.view(batch_size, -1)
        # [batch_size, out_dim]
        mean = self.fc_mean(flatten_src)
        # [batch_size, out_dim]
        logvar = self.fc_logvar(flatten_src)

        return mean, logvar


class MusicGenerator(nn.Module):
    def __init__(self, n_fft, in_dim, max_len):
        super(MusicGenerator, self).__init__()

        self.fc = nn.Linear(in_dim, 1024)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hid_dim=16, n_heads=6, pos_ff_dim=512, dropout=0.1),

            EncoderLayer(hid_dim=32, n_heads=6, pos_ff_dim=512, dropout=0.1),
            EncoderLayer(hid_dim=64, n_heads=6, pos_ff_dim=512, dropout=0.1),
            EncoderLayer(hid_dim=128, n_heads=6, pos_ff_dim=512, dropout=0.1),
        ])


    def forward(self, x): # [batch_size, in_dim=512]
        # [batch_size, 1024]
        x = self.fc(x)
        # [batch_size, max_len=64, hid_dim=16]
        x = x.view(-1, 64, 16)
        


def main():
    model = GestureEncoder(
        num_layers=6, 
        n_heads=8, 
        in_dim=69, 
        hid_dim=256, 
        out_dim=512, 
        max_len=640, 
        pos_ff_dim=512, 
        dropout=0.1
    ).to('cpu')
    
    # testing if model can output matrix with correct dimension
    src = torch.randn(20, 640, 69)
    mean, logvar = model.forward(src)

    print(mean.shape)
    print(logvar.shape)
