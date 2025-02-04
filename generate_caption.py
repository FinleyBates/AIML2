import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import nltk
import os
import json
#generates completely nonsensical captions
nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open("models/vocab.json", "r") as f:
    vocab = json.load(f)
idx_to_word = {idx: word for word, idx in vocab.items()}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

weights = VGG16_Weights.DEFAULT
vgg16 = vgg16(weights=weights)

feature_extractor = torch.nn.Sequential(*list(vgg16.children())[:-2]).to(device)
feature_extractor.load_state_dict(torch.load("models/feature_extractor_4.pth"))
feature_extractor.eval()

import torch.nn as nn

class CaptionDecoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size, num_layers=2):
        super(CaptionDecoder, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, features, hidden=None):
        outputs, hidden = self.lstm(features, hidden)
        outputs = self.fc(outputs)
        return outputs, hidden

feature_dim = 512
hidden_dim = 512
decoder = CaptionDecoder(feature_dim, hidden_dim, len(vocab)).to(device)
decoder.load_state_dict(torch.load("models/decoder_4.pth"))
decoder.eval()
temperature = 1.0

def generate_caption(image_path, feature_extractor, decoder, idx_to_word, max_caption_length=20, temperature=1.0):
    feature_extractor.eval()
    decoder.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(image_tensor)

    features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)

    captions = []
    hidden = None
    word_id = vocab['<SOS>']

    for _ in range(max_caption_length):
        input_word = torch.tensor([[word_id]], dtype=torch.long).to(device)
        output, hidden = decoder(features[:, :1, :], hidden)
        output = output.squeeze(1)
        probs = torch.nn.functional.softmax(output / temperature, dim=-1)
        word_id = torch.multinomial(probs, num_samples=1).item()

        if word_id == vocab['<EOS>']:
            break

        if word_id not in (vocab['<SOS>'], vocab['<PAD>']):
            captions.append(idx_to_word[word_id])

    return " ".join(captions)

test_image_path = "untrained_images/motorbike.jpg"
generated_caption = generate_caption(test_image_path, feature_extractor, decoder, idx_to_word)
print(f"Vocabulary size: {len(vocab)}")
print(f"Sample vocab mapping: {list(vocab.items())[:5]}")
print(f"Sample idx_to_word mapping: {list(idx_to_word.items())[:5]}")
print(f"Generated Caption: {generated_caption}")
print(f"<SOS> index: {vocab.get('<SOS>', 'Not found')}")
print(f"<EOS> index: {vocab.get('<EOS>', 'Not found')}")

with torch.no_grad():
    features = torch.rand(1, 10, 512).to(device)
    hidden = None
    input_word = torch.tensor([[vocab['<SOS>']]], dtype=torch.long).to(device)
    output, hidden = decoder(features[:, :1, :], hidden)
    probs = torch.nn.functional.softmax(output.squeeze(1), dim=-1)
    predicted_word_id = torch.argmax(probs, dim=-1).item()

print(f"Predicted Word ID: {predicted_word_id}")
print(f"Predicted Word: {idx_to_word.get(predicted_word_id, 'Unknown')}") 