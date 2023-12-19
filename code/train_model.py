import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Transformer Encoder Model


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        encoded = self.transformer_encoder(embedded)
        return self.fc(encoded.mean(dim=1))

# Transformer Decoder Model


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, src, memory):
        embedded = self.embedding(src)
        decoded = self.transformer_decoder(embedded, memory)
        return self.fc(decoded.mean(dim=1))


class CodeDataset(Dataset):
    def __init__(self, paths, function_names, vocab_size):
        self.paths = paths
        self.function_names = function_names
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # This is where you would convert paths and function names to tensor of token indices
        # For simplicity, let's assume paths and function_names are already tokenized
        return torch.tensor(self.paths[idx]), torch.tensor(self.function_names[idx])


# Example Usage
# paths = [[tokenized path 1], [tokenized path 2], ...]
# function_names = [[tokenized name 1], [tokenized name 2], ...]
dataset = CodeDataset(paths, function_names, vocab_size=10000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Model Hyperparameters
input_dim = 10000  # Size of your vocabulary
embed_dim = 128    # Size of each embedding vector
num_heads = 4      # Number of heads in the multiheadattention models
num_layers = 2     # Number of sub-encoder-layers in the encoder
output_dim = 100   # Output dimension (size of the function names vocabulary)
num_epochs = 10

# Initialize models
encoder = TransformerEncoder(
    input_dim, embed_dim, num_heads, num_layers, embed_dim)
decoder = TransformerDecoder(
    input_dim, embed_dim, num_heads, num_layers, output_dim)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

# Training Loop
for epoch in range(num_epochs):
    for paths, function_names in dataloader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Forward pass through encoder
        encoder_output = encoder(paths)

        # Forward pass through decoder
        decoder_output = decoder(function_names, encoder_output)

        # Calculate loss and backpropagate
        loss = criterion(decoder_output, function_names)
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
