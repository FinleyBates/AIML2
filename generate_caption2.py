import torch
import torchvision.models as models
from torchvision.models import VGG16_Weights
import torchvision.transforms as transforms
from PIL import Image
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.nn.functional import softmax
import matplotlib.pyplot as plt

#use gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load vocab generated from captions
with open("models/vocab.json", "r") as f:
    vocab = json.load(f)

idx_to_word = {idx: word for word, idx in vocab.items()}

#transform the images so they're the size specific to the VGG16 model
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),          #convert to tensor
    transforms.Normalize(           
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

#defines the caption decoder model, using the LSTM network
class CaptionDecoder(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size, num_layers=2): #uses multiple layers to help the network understand patterns in the data to a higher level
        super(CaptionDecoder, self).__init__()
        self.lstm = torch.nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, hidden=None):
        outputs, hidden = self.lstm(features, hidden)
        outputs = self.fc(outputs)
        return outputs, hidden

#loading predefined VGG16 feature extraction model as well as the custom decoder model
def load_model():
    feature_extractor = torch.nn.Sequential(*list(models.vgg16(weights=models.VGG16_Weights.DEFAULT).children())[:-2])
    feature_extractor.eval()
    feature_extractor.to(device)

    decoder = CaptionDecoder(feature_dim=512, hidden_dim=512, vocab_size=len(vocab))
    decoder.load_state_dict(torch.load("models/decoder_4.pth", map_location=device)) #loads custom decoder trained by training_4
    decoder.eval()
    decoder.to(device)

    return feature_extractor, decoder

def generate_caption(image_path, feature_extractor, decoder, max_caption_length=20, temperature=0.9): #temperature controlling the randomness of predictions - I found 0.9 to give me the best results
    #preprocess image
    image = Image.open(image_path).convert("RGB") #transformed to be passed through VGG16
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        features = feature_extractor(image_tensor)
        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)  #shape: (1, seq_len, feature_dim)

    hidden = None
    input_features = features[:, 0:1, :]  #using the firt feature vector as input
    captions = []

    word_id = vocab['<SOS>']  #<sos> taken as start token
    for _ in range(max_caption_length):
        input_word = torch.tensor([[word_id]], dtype=torch.long).to(device)
        output, hidden = decoder(input_features, hidden)
        output = output.squeeze(1)  #shape: (1, vocab_size)

        #applying softmax and sampling - taking the vector of logits and normalizing them so the output can be interpreted as a probability distribution
        probs = softmax(output / temperature, dim=-1)
        word_id = torch.multinomial(probs, num_samples=1).item()

        #break when <eos> token reached
        if word_id == vocab['<EOS>']:
            break

        captions.append(idx_to_word[word_id])

        #updating the input features for the next step
        input_features = features[:, 0:1, :] 

    #filtering out unwanted tokens in the caption result
    return " ".join([word for word in captions if word not in ['<SOS>', '<PAD>']])

def compute_bleu(reference_captions, generated_caption):
    reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in reference_captions]
    generated_tokens = nltk.word_tokenize(generated_caption.lower())
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing_function)

def visualize_caption(image_path, generated_caption, reference_captions): #visualizes the image with the captions
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Generated: {caption}\nReference: {', '.join(reference_captions)}")
    plt.show()

if __name__ == "__main__":
    feature_extractor, decoder = load_model()

    #image path
    image_path = "untrained_images/mountains.jpg"
    reference_captions = ["A man in a red jacket is sitting on a bench whilst cooking a meal", "A man is sitting on a bench , cooking some food ", "A man sits on a bench", "A man wearing a red jacket is sitting on a wooden bench and is cooking something in a small pot ."]

    caption = generate_caption(image_path, feature_extractor, decoder)
    bleu_score = compute_bleu(reference_captions, caption)
    visualize_caption(image_path, caption, reference_captions)
    print("Generated Caption:", caption)
    print("BLEU Score:", bleu_score)
    print("Reference Captions:", reference_captions)



    
    
