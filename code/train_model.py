import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np


# Transformer Encoder Model
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, src, memory):
        embedded = self.embedding(src)
        decoded = self.transformer_decoder(embedded, memory)
        return self.fc(decoded.mean(dim=1))

class Vocabulary:
    def __init__(self, special_tokens=None):
        self.itos = {}  # integer-to-string mapping
        self.stoi = {}  # string-to-integer mapping

        self.special_tokens = special_tokens if special_tokens else ['<PAD>', '<UNK>']

        for i, token in enumerate(self.special_tokens):
            self.itos[i] = token
            self.stoi[token] = i

    def add_token(self, token):
        if token not in self.stoi:
            index = len(self.itos)
            self.itos[index] = token
            self.stoi[token] = index

    def __len__(self):
        return len(self.itos)

    def encode(self, token_list):
        return [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in token_list]

    def decode(self, index_list):
        return [self.itos[index] for index in index_list]
    
class LabelEncoder:
    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = {}

    def add_label(self, label):
        if label not in self.label_to_index:
            index = len(self.label_to_index)
            self.label_to_index[label] = index
            self.index_to_label[index] = label

    def encode(self, label):
        return self.label_to_index[label]

    def decode(self, index):
        return self.index_to_label[index]

    def __len__(self):
        return len(self.label_to_index)


class CodeDataset(Dataset):
    def __init__(self, encoded_function_paths, function_name_indices, vocab):
        self.encoded_function_paths = encoded_function_paths
        self.function_name_indices = function_name_indices
        self.vocab = vocab

    def __len__(self):
        return len(self.encoded_function_paths)

    def __getitem__(self, idx):
        max_length = 512
        function_path_sequence = self.encoded_function_paths[idx][:max_length]
        padded_sequence = function_path_sequence + [self.vocab.stoi['<PAD>']] * (max_length - len(function_path_sequence))
        
        function_name_index = self.function_name_indices[idx]
        return torch.tensor(padded_sequence), torch.tensor(function_name_index)


def tokenize_and_encode_paths(paths_list, vocab):
    """
    Tokenizes and encodes a list of paths for each function.
    Each path is a list of elements. 
    """
    encoded_paths = []
    for paths in paths_list:  # Iterate over lists of paths for each function
        encoded_function_paths = []
        for path in paths:  # Iterate over each path
            for element in path.split('/'):  # Tokenize each element in the path
                vocab.add_token(element)  # Add element to vocabulary
                encoded_function_paths.append(vocab.stoi[element])  # Encode the element
        encoded_paths.append(encoded_function_paths)
    return encoded_paths

def calculate_metrics(preds, labels):
    preds = np.argmax(preds, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return accuracy, precision, f1


# dataset tokenizing and loading
vocab = Vocabulary(special_tokens=['<PAD>', '<UNK>'])
encoded_function_paths = tokenize_and_encode_paths(function_paths, vocab)

label_encoder = LabelEncoder()
for name in function_names:
    label_encoder.add_label(name)
encoded_function_names = [label_encoder.encode(name) for name in function_names]

dataset = CodeDataset(encoded_function_paths, encoded_function_names, vocab)

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model Hyperparameters
input_dim = 10000  # Size of your vocabulary
embed_dim = 128    # Size of each embedding vector
num_heads = 4      # Number of heads in the multiheadattention models
num_layers = 2     # Number of sub-encoder-layers in the encoder
output_dim = 100   # Output dimension (size of the function names vocabulary)
num_epochs = 10

# Initialize models
encoder = TransformerEncoder(input_dim, embed_dim, num_heads, num_layers, embed_dim)
decoder = TransformerDecoder(input_dim, embed_dim, num_heads, num_layers, output_dim)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

# Training Loop
for epoch in range(num_epochs):
    for paths, function_names in train_loader:
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

encoder.eval()
decoder.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for paths, function_names in test_loader:
        encoder_output = encoder(paths)
        decoder_output = decoder(function_names, encoder_output)
        
        all_preds.extend(decoder_output.cpu().numpy())
        all_labels.extend(function_names.cpu().numpy())

accuracy, precision, f1 = calculate_metrics(all_preds, all_labels)
