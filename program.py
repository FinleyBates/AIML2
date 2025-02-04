import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import nltk
from collections import Counter
import os
#initial attempt
print(f"Vocabulary size: {len(vocab)}")

vgg16 = models.vgg16(pretrained=True)
vgg16.eval()

#define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),          
    transforms.Normalize(           
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])


image_path = "untrained_images/mountain.jpg"
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)


feature_extractor = torch.nn.Sequential(*list(vgg16.children())[:-2])


with torch.no_grad():
    features = feature_extractor(image_tensor)

print(features.shape)

import torch.nn as nn

class CaptionDecoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size, num_layers=1):
        super(CaptionDecoder, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, features, captions):
        outputs, _ = self.lstm(features)
        outputs = self.fc(outputs)
        return outputs


feature_dim = features.shape[1]
hidden_dim = 512
vocab_size = 10000 #later changed to 5000
decoder = CaptionDecoder(feature_dim, hidden_dim, vocab_size)

import torch.optim as optim

#defininign loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(decoder.parameters(), lr=0.001)


for epoch in range(num_epochs):
    for images, captions in dataloader:
   
        with torch.no_grad():
            features = feature_extractor(images)
        
      
        outputs = decoder(features, captions)
        
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def generate_caption(feature_extractor, decoder, image_path, vocab):

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(image_tensor)
    

    captions = []
    decoder.eval()
    with torch.no_grad():
        hidden = None
        input_features = features
        for _ in range(max_caption_length):
            output, hidden = decoder.lstm(input_features, hidden)
            word_id = output.argmax(2).item()
            captions.append(vocab[word_id])
            input_features = output 
        
        return " ".join(captions)


generated_caption = generate_caption(feature_extractor, decoder, "untrained_images/mountain.jpg", vocab)
print("Generated Caption:", generated_caption)
