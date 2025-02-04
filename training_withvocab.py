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

nltk.download('punkt')
nltk.download('punkt_tab')

checkpoint_dir = "checkpoints"
latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

start_epoch = 0
start_batch = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


captions_path = "flickr8k/captions.txt"
captions_df = pd.read_csv(captions_path, names=["image", "caption"], skiprows=1)


captions_df['image'] = captions_df['image'].apply(lambda x: x.split('#')[0])

#print(captions_df.head())

captions_df['tokens'] = captions_df['caption'].apply(lambda x: nltk.word_tokenize(x.lower()))
#print(captions_df['tokens'].head())

all_tokens = [token.lower() for tokens in captions_df['tokens'] for token in tokens]
vocab = Counter(all_tokens)
vocab_size = 5000 
vocab = {word: idx for idx, (word, _) in enumerate(vocab.most_common(vocab_size), start=1)}


vocab['<PAD>'] = 0
vocab['<SOS>'] = len(vocab) 
vocab['<EOS>'] = len(vocab)


idx_to_word = {idx: word for word, idx in vocab.items()}


word_counts = Counter([word for tokens in captions_df['tokens'] for word in tokens])
word_freqs = torch.tensor([word_counts.get(word, 0) for word in vocab.keys()], dtype=torch.float)


class_weights = 1.0 / (word_freqs + 1e-5) 
class_weights = class_weights / class_weights.sum() 


print(f"Vocabulary size: {len(vocab)}")

print(torch.cuda.is_available()) 
print(torch.cuda.device_count())  
print(torch.cuda.get_device_name(0))  


weights = VGG16_Weights.DEFAULT
vgg16 = vgg16(weights=weights)


import torch.nn as nn

class CaptionDecoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size, num_layers=2):
        super(CaptionDecoder, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5) 
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, features, captions):
        outputs, _ = self.lstm(features)
        outputs = self.fc(outputs)
        return outputs


feature_dim = 512 
hidden_dim = 512
decoder = CaptionDecoder(feature_dim, hidden_dim, len(vocab))


criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) 
optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)



feature_extractor = torch.nn.Sequential(*list(vgg16.children())[:-2])
feature_extractor.eval()

feature_extractor = feature_extractor.to(device)
decoder = decoder.to(device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),          
    transforms.Normalize(         
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

if os.path.exists(latest_checkpoint_path):
    print(f"Loading checkpoint from {latest_checkpoint_path}...")
    checkpoint = torch.load(latest_checkpoint_path)

    feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_batch = checkpoint['batch']
    print(f"Resumed from epoch {start_epoch}, batch {start_batch}")
else:
    print("No checkpoint found. Starting from scratch.")


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

# Define collate_fn
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0) 
    captions = pad_sequence(captions, batch_first=True, padding_value=vocab['<PAD>'])
    return images, captions

image_dir = "flickr8k/Images"
dataset = Flickr8kDataset(captions_df, image_dir, vocab, transform=transform)


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

def save_checkpoint(epoch, batch, feature_extractor, decoder, optimizer, loss, vocab, idx_to_word, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True) 
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}_batch{batch}.pth")
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

    torch.save({
        'epoch': epoch,
        'batch': batch,
        'feature_extractor_state_dict': feature_extractor.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'vocab': vocab, 
        'idx_to_word': idx_to_word
    }, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

   
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'feature_extractor_state_dict': feature_extractor.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'vocab': vocab, 
        'idx_to_word': idx_to_word 
    }, latest_checkpoint_path)
    print(f"Latest checkpoint updated: {latest_checkpoint_path}")


def save_vocab(vocab, vocab_path="models/vocab.json"):
    import json
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    print(f"Vocabulary saved to {vocab_path}")


num_epochs = 10

for epoch in range(start_epoch, num_epochs):
    for batch_idx, (images, captions) in enumerate(dataloader):
    
        if epoch == start_epoch and batch_idx < start_batch:
            continue

        images = images.to(device)
        captions = captions.to(device)

        print(f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}")
        print("Batch Images Shape:", images.shape)
        print("Batch Captions Shape:", captions.shape)

 
        with torch.no_grad():
            features = feature_extractor(images)
            batch_size, channels, height, width = features.shape

          
            features = features.view(batch_size, channels, -1).permute(0, 2, 1)
            sequence_length = captions.size(1) 
            features = features[:, :sequence_length, :]


        outputs = decoder(features, captions)


        loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #save checkpoint
    save_checkpoint(epoch + 1, "end", feature_extractor, decoder, optimizer, loss.item(), vocab, idx_to_word)

    #save vocab to seperate file
    save_vocab(vocab, vocab_path="models/vocab.json")

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


torch.save(feature_extractor.state_dict(), "models/feature_extractor_trained_3.pth")

torch.save(decoder.state_dict(), "models/decoder_trained_3.pth")

torch.save(optimizer.state_dict(), "models/optimizer_state_3.pth")

print("Models and optimizer state saved successfully.")
