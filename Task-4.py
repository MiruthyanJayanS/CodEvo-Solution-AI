import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Step 1: Load and Preprocess the Data
with open("shakespeare.txt", 'r') as f:
    text = f.read()

# Create a character-to-index and index-to-character mapping
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Convert the text into integer indices
text_as_int = np.array([char_to_idx[c] for c in text])

# Set sequence length and batch size
SEQ_LENGTH = 100
BATCH_SIZE = 64

# Create input-output sequences
def create_sequences(text, seq_length):
    inputs = []
    targets = []
    for i in range(0, len(text) - seq_length):
        inputs.append(text[i:i+seq_length])
        targets.append(text[i+seq_length])
    return np.array(inputs), np.array(targets)

inputs, targets = create_sequences(text_as_int, SEQ_LENGTH)

# Create a PyTorch dataset
class TextDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

dataset = TextDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Step 2: Define Transformer-based Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(SEQ_LENGTH, d_model)
        self.transformer = nn.Transformer(d_model, num_heads, num_layers, num_layers, dropout=dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        # Add positional encoding to the input
        seq_length = src.shape[1]
        pos = torch.arange(0, seq_length).unsqueeze(0).repeat(src.size(0), 1).to(src.device)
        embedded = self.embedding(src) + self.pos_encoder(pos)
        
        # Transformer expects (sequence_length, batch_size, embedding_dim)
        embedded = embedded.transpose(0, 1)
        transformer_output = self.transformer(embedded, embedded)
        output = self.fc(transformer_output[-1])  # Take the output of the last token
        return output

# Step 3: Training Loop
# Model parameters
VOCAB_SIZE = len(chars)
D_MODEL = 128
NUM_HEADS = 8
NUM_LAYERS = 4
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def train(model, dataloader, optimizer, loss_fn, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Train the model
train(model, dataloader, optimizer, loss_fn, EPOCHS)

# Save model checkpoints
torch.save(model.state_dict(), "transformer_text_gen.pth")

# Step 4: Generate Text
def generate_text(model, start_text, gen_length):
    model.eval()
    input_eval = [char_to_idx[s] for s in start_text]
    input_eval = torch.tensor(input_eval, dtype=torch.long).unsqueeze(0).to(device)
    
    generated_text = start_text
    for _ in range(gen_length):
        with torch.no_grad():
            output = model(input_eval)
            prediction = torch.softmax(output, dim=-1)
            next_char_idx = torch.multinomial(prediction, num_samples=1).item()
            next_char = idx_to_char[next_char_idx]
            generated_text += next_char
            
            # Prepare the input for the next step
            input_eval = torch.cat([input_eval[:, 1:], torch.tensor([[next_char_idx]], dtype=torch.long).to(device)], dim=1)
    
    return generated_text

# Generate a text sample
start_text = "To be, or not to be, "
generated = generate_text(model, start_text, 500)
print(generated)
