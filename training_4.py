import torch #import libraries such as pyrorch and torchvision - I chose to use pytorch due to its beginner friendly ease of use aswell as allowing for GPU usage in training
import torch.nn as nn # pytorch is also already integrated with VGG16 so that was an easy choice of pre-trained encoder
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #attempt to use gpu for faster trinaing
print(f"Using device: {device}")

#paths 
checkpoint_dir = "checkpoints"
latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
captions_path = "flickr8k/captions.txt"
image_dir = "flickr8k/Images"

#data prep
captions_df = pd.read_csv(captions_path, names=["image", "caption"], skiprows=1)#first row has non-relevent info
captions_df['image'] = captions_df['image'].apply(lambda x: x.split('#')[0]) #split the line 
captions_df['tokens'] = captions_df['caption'].apply(lambda x: nltk.word_tokenize(x.lower()))

#save vocab
all_tokens = [token.lower() for tokens in captions_df['tokens'] for token in tokens]
vocab = Counter(all_tokens)
vocab_size = 5000  #limits vocabulary size to the top 5000 most common words - this is to convert the words in to numbers for the model to interpret
vocab = {word: idx for idx, (word, _) in enumerate(vocab.most_common(vocab_size), start=1)}#putting the vocab in order

#adding special tokens 
vocab['<PAD>'] = 0
vocab['<SOS>'] = len(vocab) #start of sentence token 
vocab['<EOS>'] = len(vocab) #end of sentence token
idx_to_word = {idx: word for word, idx in vocab.items()}

#using the VGG16 pretrained feature extraction model
weights = VGG16_Weights.DEFAULT
vgg16 = vgg16(weights=weights) 
feature_extractor = torch.nn.Sequential(*list(vgg16.children())[:-2]).to(device) #we only use the earlier layers as we don't need the classification layers the VGG16 model offers
feature_extractor.eval() #set the extractor to eval mode

#custom decoder model - uses LSTM to process image features and generate words sequentially. After finished, a fully connected layer will predict the next word from the vocabulary
class CaptionDecoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size, num_layers=2):
        super(CaptionDecoder, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5) #dropout used to prevent overfitting, causes noise by adding randomness in
        self.fc = nn.Linear(hidden_dim, vocab_size)                                             #turning off neruons. this prevents the model from getting stuck in a groove
    
    def forward(self, features, captions):
        embeddings, _ = self.lstm(features)
        outputs = self.fc(embeddings)
        return outputs

feature_dim = 512
hidden_dim = 512
decoder = CaptionDecoder(feature_dim, hidden_dim, len(vocab)).to(device)

#gives a different level of importance to each word during training, initialises all weights to 1 and <pad> to 0
class_weights = torch.tensor([1.0] * len(vocab), dtype=torch.float).to(device)
class_weights[vocab['<PAD>']] = 0.0

#defining the loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=vocab['<PAD>']) #defining loss function criterion which combines logsoftmax and negative log 
                                                                                   #likelihood to calculate the difference between predicted and target words
optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001) #updates the models parameters to minimize loss

#flickr8k dataset
class Flickr8kDataset(torch.utils.data.Dataset):
    def __init__(self, captions_df, image_dir, vocab, transform=None):
        self.captions_df = captions_df
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform #resized to fit the model

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

#defining number of epochs for the training loop
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (images, captions) in enumerate(dataloader): #dataloader provides batches of images and captions from the dataset (32 at a time)
        images, captions = images.to(device), captions.to(device) #moves images and caption tensors to the cpu(using cuda)

        with torch.no_grad(): #VGG16 feature extractor used to convert input features
            features = feature_extractor(images)
            features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1) #reshaped to match decoder

        
        decoder_inputs = captions[:, :-1]  #input is all but the last token
        decoder_targets = captions[:, 1:]  #target is all but the first token

        #truncate features to match decoder inputs
        features = features[:, :decoder_inputs.size(1), :]

        #decoder forward pass takes image features and looks at captions to predict the next word for each step
        outputs = decoder(features, decoder_inputs)
        outputs = outputs.view(-1, len(vocab))
        decoder_targets = decoder_targets.contiguous().view(-1)

        # uses criterion to calculate difference as defined earlier
        loss = criterion(outputs, decoder_targets)

        optimizer.zero_grad() #clear gradients from prvious step
        loss.backward() #calculates gradients of loss
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0) #limits gradient magnitude to account for outliers
        optimizer.step() #updates model parameters

        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss/len(dataloader):.4f}")

#save model weights
torch.save(feature_extractor.state_dict(), "models/feature_extractor_4.pth")
torch.save(decoder.state_dict(), "models/decoder_4.pth")
torch.save(optimizer.state_dict(), "models/optimizer_4.pth")

print("Training complete and models saved.")
