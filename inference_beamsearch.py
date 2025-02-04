import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import nltk
import pandas as pd
from collections import Counter
import os
import torch.nn.functional as F

#ensure nltk datasets are downloaded
# nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

captions_path = "flickr8k/captions.txt"
captions_df = pd.read_csv(captions_path, names=["image", "caption"], skiprows=1)

captions_df['tokens'] = captions_df['caption'].apply(lambda x: nltk.word_tokenize(x.lower()))
all_tokens = [token for tokens in captions_df['tokens'] for token in tokens]
vocab_size = 5000  #limit vocabulary size
vocab = {word: idx for idx, (word, _) in enumerate(Counter(all_tokens).most_common(vocab_size), start=1)}

vocab['<PAD>'] = 0
vocab['<SOS>'] = len(vocab)
vocab['<EOS>'] = len(vocab)
idx_to_word = {idx: word for word, idx in vocab.items()}  #reverse vocabulary mapping

print(f"Vocabulary size: {len(vocab)}")

# Define preprocessing transformations
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
# print("First layer weights:", decoder.fc.weight)


def generate_caption_with_beam_search(feature_extractor, decoder, image_path, idx_to_word, vocab, beam_width=5, max_caption_length=30):
    feature_extractor.eval()
    decoder.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(image_tensor)
    features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)

    #initialize the beam search
    sequences = [([[vocab['<SOS>']], 0.0])] #start with <eos> token and score of 0.0
    hidden = None

    for step in range(max_caption_length):
        all_candidates = []
        for seq, score in sequences:
            print(f"Step {step}, Sequence: {seq}, Decoded: {[idx_to_word[token] for token in seq]}, Score: {score}")

            if seq[-1] == vocab['<EOS>']:
                if len(seq) < 5:  
                    continue
                all_candidates.append((seq, score))
                continue

            #using extracted feature for the first step and decoder for following
            input_features = features if len(seq) == 1 else decoder.fc.weight[seq[-1]].unsqueeze(0).unsqueeze(0).to(device)

            #pass through the decodder
            output, hidden = decoder(input_features, hidden)
            probabilities = F.softmax(output.squeeze(0), dim=-1)

            # Debugging: Print top-k predictions
            topk_probs, topk_indices = probabilities.topk(beam_width)
            print(f"Top-k probabilities: {topk_probs[0].tolist()}")
            print(f"Top-k words: {[idx_to_word[idx] for idx in topk_indices[0].tolist()]}")

            for i in range(beam_width):
                token_id = topk_indices[0, i].item()
                candidate = seq + [token_id]

                #compute scores
                candidate_score = score + torch.log(topk_probs[0, i]).item()
                if len(candidate) < 5 and token_id == vocab['<EOS>']:
                    candidate_score -= 1.0  #penalize early <eos>
                if token_id in [vocab['<SOS>'], vocab['<PAD>']]:
                    candidate_score -= 1.0 #penalize invalid

                #normalize score
                candidate_score /= len(candidate)

                all_candidates.append((candidate, candidate_score))

        #sort and prune to beam width 
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        #early stop if all sequences end with <EOS>
        if all(seq[-1] == vocab['<EOS>'] for seq, _ in sequences):
            break

    #select highest scoring sequence
    final_seq = sequences[0][0]

    return ' '.join([idx_to_word[token] for token in final_seq if token not in [vocab['<SOS>'], vocab['<PAD>'], vocab['<EOS>']]])


test_image_path = "97DF5226-04F3-4A6D-807B-5D0F9A11879C_1024x1024.jpg"

image = Image.open(test_image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    features = feature_extractor(image_tensor)
    print(f"Extracted features shape: {features.shape}")

generated_caption = generate_caption_with_beam_search(feature_extractor, decoder, test_image_path, idx_to_word, vocab)
print(f"Generated Caption: {generated_caption}")

#print(f"Vocabulary sample: {list(vocab.items())[:10]}")
#print(f"Reverse vocabulary sample: {list(idx_to_word.items())[:10]}")

# Test decoder with random inputs
#dummy_input = torch.randn(1, 1, feature_dim).to(device)  # Batch size 1, sequence length 1, feature_dim
#output, _ = decoder(dummy_input)
#print(f"Decoder output shape: {output.shape}")  # Should be [1, 1, vocab_size]
#print(f"Top predicted token: {output.argmax(dim=-1).item()}")  # Check token ID
#print(f"Top predicted word: {idx_to_word[output.argmax(dim=-1).item()]}")
