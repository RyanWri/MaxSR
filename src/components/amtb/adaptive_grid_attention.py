import torch
import torch.nn as nn

class GridSelfAttention(nn.Module):
    def __init__(self, in_features, heads=8):
        super(GridSelfAttention, self).__init__()
        self.heads = heads
        self.in_features = in_features // heads

        self.query = nn.Linear(self.in_features, self.in_features)
        self.key = nn.Linear(self.in_features, self.in_features)
        self.value = nn.Linear(self.in_features, self.in_features)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, num_patches, features = x.shape
        x = x.view(batch_size, num_patches, self.heads, self.in_features)

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.in_features ** 0.5)
        attention = self.softmax(attention_scores)

        out = torch.matmul(attention, values)
        out = out.view(batch_size, num_patches, -1)
        return out

class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class AdaptiveGridAttention(nn.Module):
    def __init__(self, in_features, hidden_features, heads=8):
        super(AdaptiveGridAttention, self).__init__()
        self.grid_self_attention = GridSelfAttention(in_features, heads=heads)
        self.ffn = FeedForwardNetwork(in_features, hidden_features)

    def forward(self, x):
        # Grid self-attention part
        grid_attention_output = self.grid_self_attention(x)
        x = grid_attention_output + x  # Add the input to the output of grid self-attention

        # Feed-forward network part
        ffn_output = self.ffn(x)
        x = ffn_output + x  # Add the input to the output of FFN

        return x

# Example usage
in_features = 128  # Number of features per patch
hidden_features = 512  # Hidden layer size in FFN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adaptive_grid_attention = AdaptiveGridAttention(in_features, hidden_features)
adaptive_grid_attention = adaptive_grid_attention.to(device)  # Ensure it's on the correct device

# Assuming 'output' from the previous block
output = torch.randn(128, 64, 128).to(device)  # Simulating input
final_grid_output = adaptive_grid_attention(output)
print("Final grid attention output shape:", final_grid_output.shape)
