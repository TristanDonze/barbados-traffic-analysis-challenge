import torch
import torch.nn as nn
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
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
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

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        scores = self.attn(x) 
        weights = torch.softmax(scores, dim=1)
        context = x * weights
        context = torch.sum(context, dim=1)
        
        return context, weights

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


class MultiHeadSequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_channels, seq_len, prediction_steps, num_streams, num_classes, dropout=0.2,):
        super(MultiHeadSequenceModel, self).__init__()

        self.backbone = TemporalConvNet(
            input_dim, hidden_channels, kernel_size=3, dropout=dropout)

        self.prediction_steps = prediction_steps
        self.num_classes = num_classes
        self.num_streams = num_streams

        final_dim = hidden_channels[-1]

        self.heads = nn.ModuleList([
            nn.Linear(final_dim, prediction_steps * num_classes)
            for _ in range(num_streams)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.backbone(x)
        context = out[:, :, -1]

        outputs = []
        for head in self.heads:
            head_out = head(context)
            head_out = head_out.view(-1,
                                     self.prediction_steps, self.num_classes)
            outputs.append(head_out)

        return torch.stack(outputs, dim=1)


class UniversalTrafficModel(nn.Module):
    def __init__(self, input_dim, hidden_channels, seq_len, prediction_steps, num_classes=4, dropout=0.2, use_attention=False, aggregate=True):
        super(UniversalTrafficModel, self).__init__()
        
        self.aggregate = aggregate
        
        if self.aggregate:
            self.num_streams = 4 
        else:
            self.num_streams = 1
            
        self.prediction_steps = prediction_steps
        self.num_classes = num_classes

        self.backbone = TemporalConvNet(input_dim, hidden_channels, kernel_size=3, dropout=dropout)
        
        
        final_dim = hidden_channels[-1]
        
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = TemporalAttention(final_dim)
            
        self.heads = nn.ModuleList([
            nn.Linear(final_dim, prediction_steps * num_classes) 
            for _ in range(self.num_streams)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.backbone(x)
        
        if self.use_attention:
            context, attn_weights = self.attention(out)
        else:
            context = out[:, :, -1]

        outputs = []
        for head in self.heads:
            head_out = head(context)
            head_out = head_out.view(-1, self.prediction_steps, self.num_classes)
            outputs.append(head_out)

        return torch.stack(outputs, dim=1)
    

if __name__ == "__main__":
    
    c1d = Chomp1d(2)
    X = torch.randn(3, 5, 10)
    print("Input shape:", X.shape)
    Y = c1d(X)
    print("Output shape after Chomp1d:", Y.shape)
    # SEQ_LEN = 15
    # HORIZON = 7
    # EMBARGO = 0
    # BATCH_SIZE = 32

    # DEVICE = torch.device("cuda" if torch.cuda.is_available(
    # ) else "mps" if torch.backends.mps.is_available() else "cpu")

    # # model = MultiHeadSequenceModel(
    # #     input_dim=328,
    # #     hidden_channels=[64, 64, 64, 64],
    # #     seq_len=SEQ_LEN,
    # #     prediction_steps=HORIZON,
    # #     num_streams=4,
    # #     num_classes=4,
    # #     dropout=0.3
    # # ).to(DEVICE)
    
    # model_1 = UniversalTrafficModel(
    #     input_dim=82,
    #     hidden_channels=[64, 64, 64, 64],
    #     seq_len=SEQ_LEN,
    #     prediction_steps=HORIZON,
    #     num_classes=4,
    #     dropout=0.3,
    #     aggregate=False
    # ).to(DEVICE)
    
    # model_2 = UniversalTrafficModel(
    #     input_dim=328,
    #     hidden_channels=[64, 64, 64, 64],
    #     seq_len=SEQ_LEN,
    #     prediction_steps=HORIZON,
    #     num_classes=4,
    #     dropout=0.3,
    #     aggregate=True
    # ).to(DEVICE)

    # X_1 = torch.randn(BATCH_SIZE, SEQ_LEN, 82).to(DEVICE)
    # outputs_1 = model_1(X_1)
    # print(f"Output shape (individual): {outputs_1.shape}")
    # X_2 = torch.randn(BATCH_SIZE, SEQ_LEN, 328).to(DEVICE)
    # outputs_2 = model_2(X_2)
    # print(f"Output shape (aggregate): {outputs_2.shape}")
