import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import nltk
import pandas as pd
from collections import Counter
import os
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

captions_path = "flickr8k/captions.txt"
captions_df = pd.read_csv(captions_path, names=["image", "caption"], skiprows=1)

captions_df['tokens'] = captions_df['caption'].apply(lambda x: nltk.word_tokenize(x.lower()))
all_tokens = [token for tokens in captions_df['tokens'] for token in tokens]
vocab_size = 5000
vocab = {word: idx for idx, (word, _) in enumerate(Counter(all_tokens).most_common(vocab_size), start=1)}

vocab['<PAD>'] = 0
vocab['<SOS>'] = len(vocab)
vocab['<EOS>'] = len(vocab)
idx_to_word = {idx: word for word, idx in vocab.items()}

print(f"Vocabulary size: {len(vocab)}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

weights = VGG16_Weights.DEFAULT
vgg16 = vgg16(weights=weights)

feature_extractor = torch.nn.Sequential(*list(vgg16.children())[:-2]).to(device).eval()

class CaptionDecoder(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size, num_layers=1):
        super(CaptionDecoder, self).__init__()
        self.lstm = torch.nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, hidden=None):
        outputs, hidden = self.lstm(features, hidden)
        outputs = self.fc(outputs)
        return outputs, hidden

feature_dim = 512
hidden_dim = 512
decoder = CaptionDecoder(feature_dim, hidden_dim, len(vocab)).to(device)

feature_extractor.load_state_dict(torch.load("models/feature_extractor_trained.pth", weights_only=True))
decoder.load_state_dict(torch.load("models/decoder_trained.pth", weights_only=True))

print("Models loaded successfully!")

def generate_caption_with_sampling(feature_extractor, decoder, image_path, idx_to_word, vocab, max_caption_length=20):
    feature_extractor.eval()
    decoder.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(image_tensor)

    features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)

    captions = []
    hidden = None
    input_features = features[:, 0:1, :]

    word_id = vocab['<SOS>']

    for _ in range(max_caption_length):
        output, hidden = decoder(input_features, hidden)
        output = output.squeeze(1)

        probabilities = F.softmax(output, dim=-1)

        word_id = torch.multinomial(probabilities, 1).item()

        if word_id == vocab['<EOS>']:
            break

        captions.append(idx_to_word[word_id])

        input_features = features[:, 0:1, :]

    return " ".join([word for word in captions if word not in ['<SOS>', '<PAD>']])

test_image_path = "97DF5226-04F3-4A6D-807B-5D0F9A11879C_1024x1024.jpg"
generated_caption = generate_caption_with_sampling(feature_extractor, decoder, test_image_path, idx_to_word, vocab)
print(f"Generated Caption: {generated_caption}")
