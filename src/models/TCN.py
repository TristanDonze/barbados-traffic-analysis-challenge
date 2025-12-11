import torch
import torch.nn as nn
import math
from torch.nn.utils.parametrizations import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :].unsqueeze(1)

class UniversalTrafficModel(nn.Module):
    def __init__(self, input_dim, hidden_channels, seq_len, prediction_steps, num_classes=4, dropout=0.2, aggregate=True, use_attention=True):
        super(UniversalTrafficModel, self).__init__()
        self.aggregate = aggregate
        self.num_streams = 4 if self.aggregate else 1
        self.prediction_steps = prediction_steps
        self.use_attention = use_attention
        
        self.backbone = TemporalConvNet(input_dim, hidden_channels, kernel_size=3, dropout=dropout)
        final_dim = hidden_channels[-1]
        
        if self.use_attention:
            self.pos_encoder = PositionalEncoding(final_dim, max_len=seq_len)
            self.mha = nn.MultiheadAttention(embed_dim=final_dim, num_heads=4, dropout=dropout)
            
            head_input_dim = final_dim * 2 
            
            self.attn_pool = nn.Sequential(
                nn.Linear(final_dim, final_dim),
                nn.Tanh(),
                nn.Linear(final_dim, 1)
            )
        else:
            head_input_dim = final_dim
    
        # self.layernorm = nn.LayerNorm(head_input_dim)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, head_input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_input_dim // 2, prediction_steps * num_classes)
            )
            for _ in range(self.num_streams)
        ])

    def forward(self, x):
        x_tcn = x.transpose(1, 2)
        out = self.backbone(x_tcn)
        
        last_step = out[:, :, -1]
        
        if self.use_attention:
            out_permuted = out.permute(2, 0, 1)
            out_permuted = self.pos_encoder(out_permuted)
            
            attn_output, attn_weights = self.mha(out_permuted, out_permuted, out_permuted)
            
            attn_seq = attn_output.permute(1, 0, 2) 
            scores = self.attn_pool(attn_seq)
            weights = torch.softmax(scores.squeeze(-1), dim=1).unsqueeze(-1)
            context = (attn_seq * weights).sum(dim=1)
            combined = torch.cat([last_step, context], dim=1)
        else:
            combined = last_step
        # combined = self.layernorm(combined)

        outputs = []
        for head in self.heads:
            head_out = head(combined)
            head_out = head_out.view(-1, self.prediction_steps, 4)
            outputs.append(head_out)

        return torch.stack(outputs, dim=1)