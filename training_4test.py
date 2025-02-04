import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import pandas as pd
import nltk
from collections import Counter
import os
from torch.cuda.amp import GradScaler, autocast

nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

checkpoint_dir = "checkpoints"
latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
captions_path = "flickr8k/captions.txt"
image_dir = "flickr8k/Images"

captions_df = pd.read_csv(captions_path, names=["image", "caption"], skiprows=1)
captions_df['image'] = captions_df['image'].apply(lambda x: x.split('#')[0])
captions_df['tokens'] = captions_df['caption'].apply(lambda x: nltk.word_tokenize(x.lower()))

all_tokens = [token.lower() for tokens in captions_df['tokens'] for token in tokens]
vocab = Counter(all_tokens)
vocab_size = 5000 
vocab = {word: idx for idx, (word, _) in enumerate(vocab.most_common(vocab_size), start=1)}

vocab['<PAD>'] = 0
vocab['<SOS>'] = len(vocab)
vocab['<EOS>'] = len(vocab)
idx_to_word = {idx: word for word, idx in vocab.items()}

weights = VGG16_Weights.DEFAULT
vgg16 = vgg16(weights=weights)
feature_extractor = torch.nn.Sequential(*list(vgg16.children())[:-2]).to(device)
feature_extractor.eval()

#training with attention mechanism
class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, features, hidden_state):
        hidden_state = hidden_state.unsqueeze(1).repeat(1, features.size(1), 1)  # (batch, seq_len, hidden_dim)
        energy = torch.tanh(self.attention(torch.cat((features, hidden_state), dim=2)))
        attention_weights = torch.nn.functional.softmax(self.v(energy).squeeze(2), dim=1)  # (batch, seq_len)
        context = (features * attention_weights.unsqueeze(2)).sum(dim=1)  # (batch, feature_dim)
        return context, attention_weights

#decoder model with attention
class CaptionDecoderWithAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size, num_layers=1):
        super(CaptionDecoderWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attention = Attention(feature_dim, hidden_dim)
        self.lstm = nn.LSTM(feature_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        batch_size = features.size(0)
        hidden_state = torch.zeros(1, batch_size, hidden_dim).to(device)
        cell_state = torch.zeros(1, batch_size, hidden_dim).to(device)


        embeddings = self.embedding(captions)

        outputs = []
        for t in range(embeddings.size(1)):  
            context, _ = self.attention(features, hidden_state[-1])  #applies attention
            lstm_input = torch.cat((context.unsqueeze(1), embeddings[:, t, :].unsqueeze(1)), dim=2)
            output, (hidden_state, cell_state) = self.lstm(lstm_input, (hidden_state, cell_state))
            outputs.append(self.fc(output))

        outputs = torch.cat(outputs, dim=1)
        return outputs

feature_dim = 512
hidden_dim = 512
decoder = CaptionDecoderWithAttention(feature_dim, hidden_dim, len(vocab)).to(device)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.vocab_size = vocab_size

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist += self.smoothing / self.vocab_size
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

criterion = LabelSmoothingLoss(len(vocab), smoothing=0.1).to(device)
optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

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
        caption = row['caption']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        tokens = ['<SOS>'] + nltk.word_tokenize(caption.lower()) + ['<EOS>']
        caption_encoded = [self.vocab.get(token, self.vocab['<PAD>']) for token in tokens]
        return image, torch.tensor(caption_encoded)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = Flickr8kDataset(captions_df, image_dir, vocab, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: (
    torch.stack([i[0] for i in x]), pad_sequence([i[1] for i in x], batch_first=True, padding_value=vocab['<PAD>'])
))

num_epochs = 10
scaler = torch.amp.GradScaler()

print(vocab)
print(captions_df.head())


for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (images, captions) in enumerate(dataloader):
        images, captions = images.to(device), captions.to(device)

        with torch.no_grad():
            features = feature_extractor(images)
            features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)

        decoder_inputs = captions[:, :-1] 
        decoder_targets = captions[:, 1:] 

        with torch.amp.autocast(device_type='cuda'):
            outputs = decoder(features, decoder_inputs)
            outputs = outputs.view(-1, len(vocab))
            decoder_targets = decoder_targets.contiguous().view(-1)
            loss = criterion(outputs, decoder_targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss/len(dataloader):.4f}")

torch.save(feature_extractor.state_dict(), "models/feature_extractor_attention.pth")
torch.save(decoder.state_dict(), "models/decoder_attention.pth")
torch.save(optimizer.state_dict(), "models/optimizer_attention.pth")

print("Training complete and models saved.")
