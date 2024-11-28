import torch
import torch.nn as nn

class TextCNNBilstmWithAttention(nn.Module):
    def __init__(self, num_classes, input_dim, num_filters, filter_sizes, hidden_size, num_layers, dropout=0.5, num_heads=5):
        super(TextCNNBilstmWithAttention, self).__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_sizes = [f for f in filter_sizes if f <= input_dim]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (filter_size, input_dim), padding=(filter_size // 2, 0)) for filter_size in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(input_size=num_filters * len(filter_sizes), 
                              hidden_size=hidden_size, 
                              num_layers=num_layers, 
                              batch_first=True, 
                              bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # shape: (batch_size, 1, seq_len, input_dim)
        x = [conv(x).squeeze(3) for conv in self.convs]  # list of (batch_size, num_filters, seq_len)
        x = [torch.relu(item) for item in x]  # list of (batch_size, num_filters, seq_len)
        min_length = min([item.size(2) for item in x])
        x = [item[:, :, :min_length] for item in x]  # trim to the minimum length
        x = torch.cat(x, 1)  # shape: (batch_size, num_filters * len(filter_sizes), min_length)
        x = x.permute(0, 2, 1)  # shape: (batch_size, min_length, num_filters * len(filter_sizes))
        x = self.dropout(x)
        x, _ = self.bilstm(x)  # shape: (batch_size, min_length, hidden_size * 2)
        x, attention_weights = self.attention(x, x, x)  # shape: (batch_size, min_length, hidden_size * 2), (batch_size, min_length, min_length)
        attention_weights = attention_weights.mean(dim=1)  # average over heads, shape: (batch_size, 1, min_length)
        x = torch.mean(x, dim=1)  # aggregate over the sequence length dimension
        out = self.fc(x)  # shape: (batch_size, num_classes)
        return out, attention_weights


