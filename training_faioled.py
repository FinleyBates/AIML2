import torch
import torchvision.models as models
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import pandas as pd
import nltk
from collections import Counter
import os

# Ensure necessary nltk datasets are available
nltk.download('punkt')

# Checkpoint directory
checkpoint_dir = "checkpoints"
latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

# Models directory
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Starting epoch and batch
start_epoch = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load captions file
captions_path = "flickr8k/captions.txt"
captions_df = pd.read_csv(captions_path, names=["image", "caption"], skiprows=1)
captions_df['image'] = captions_df['image'].apply(lambda x: x.split('#')[0])
captions_df['tokens'] = captions_df['caption'].apply(lambda x: nltk.word_tokenize(x.lower()))

# Build vocabulary
all_tokens = [token for tokens in captions_df['tokens'] for token in tokens]
vocab_size = 5000
vocab = {word: idx for idx, (word, _) in enumerate(Counter(all_tokens).most_common(vocab_size), start=1)}
vocab['<PAD>'] = 0
vocab['<SOS>'] = len(vocab)
vocab['<EOS>'] = len(vocab)
idx_to_word = {idx: word for word, idx in vocab.items()}

# Print vocabulary size
print(f"Vocabulary size: {len(vocab)}")

# Load pre-trained VGG16 model
weights = VGG16_Weights.DEFAULT
vgg16 = vgg16(weights=weights)

# Feature extractor
feature_extractor = torch.nn.Sequential(*list(vgg16.children())[:-2]).to(device).eval()

# Define decoder model
class CaptionDecoder(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size, num_layers=1):
        super(CaptionDecoder, self).__init__()
        self.lstm = torch.nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions=None):
        outputs, hidden = self.lstm(features)
        outputs = self.fc(outputs)
        return outputs, hidden

feature_dim = 512
hidden_dim = 512
decoder = CaptionDecoder(feature_dim, hidden_dim, len(vocab)).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

# Load checkpoint
if os.path.exists(latest_checkpoint_path):
    print(f"Loading checkpoint from {latest_checkpoint_path}...")
    checkpoint = torch.load(latest_checkpoint_path)
    feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resumed from epoch {start_epoch}.")
else:
    print("No checkpoint found. Starting from scratch.")

# Dataset and dataloader setup
class Flickr8kDataset(torch.utils.data.Dataset):
    def __init__(self, captions_df, image_dir, vocab, transform=None):
        self.captions_df = captions_df
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        row = self.captions_df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        tokens = ['<SOS>'] + nltk.word_tokenize(row['caption'].lower()) + ['<EOS>']
        caption_encoded = [self.vocab.get(token, self.vocab['<PAD>']) for token in tokens]
        return image, torch.tensor(caption_encoded)

    @staticmethod
    def collate_fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images, 0)
        captions = pad_sequence(captions, batch_first=True, padding_value=vocab['<PAD>'])
        return images, captions

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_dir = "flickr8k/Images"
dataset = Flickr8kDataset(captions_df, image_dir, vocab, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=Flickr8kDataset.collate_fn)

# Save checkpoint
def save_checkpoint(epoch, feature_extractor, decoder, optimizer, loss):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'feature_extractor_state_dict': feature_extractor.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# Training loop
num_epochs = 10
for epoch in range(start_epoch, num_epochs):
    # Save a checkpoint at the start of each epoch
    save_checkpoint(epoch + 1, feature_extractor, decoder, optimizer, 0.0)  # Placeholder loss
    print(f"Checkpoint saved at the start of epoch {epoch + 1}.")

    feature_extractor.train()
    decoder.train()

    for batch_idx, (images, captions) in enumerate(dataloader):
        # Move data to device
        images, captions = images.to(device), captions.to(device)

        # Forward pass
        features = feature_extractor(images)  # Extract features
        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)  # Reshape for LSTM
        outputs, _ = decoder(features)  # Decoder outputs

        # Reshape outputs and captions
        batch_size, seq_len, vocab_size = outputs.shape
        outputs = outputs.view(-1, vocab_size)  # Flatten to (batch_size * seq_len, vocab_size)
        captions = captions.view(-1)  # Flatten to (batch_size * seq_len)

        # Compute loss
        valid_indices = captions != vocab['<PAD>']  # Ignore <PAD> tokens in the loss computation
        loss = criterion(outputs[valid_indices], captions[valid_indices])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print batch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")


# Save final models
torch.save(feature_extractor.state_dict(), os.path.join(models_dir, "feature_extractor_trained.pth"))
torch.save(decoder.state_dict(), os.path.join(models_dir, "decoder_trained.pth"))
torch.save(optimizer.state_dict(), os.path.join(models_dir, "optimizer_state.pth"))
print("Final models saved to the models directory.")
