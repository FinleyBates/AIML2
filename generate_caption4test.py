import torch
import torchvision.models as models
from torchvision.models import VGG16_Weights
import torchvision.transforms as transforms
from PIL import Image
import json
from torch.nn.functional import softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# attempted to use an attention mechanism to focus attention on relevant parts of the feature map
class Attention(torch.nn.Module): 
    def __init__(self, feature_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attention = torch.nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.v = torch.nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, features, hidden_state): #takes fetures and hidden state of the lstm as the input 
        hidden_state = hidden_state.unsqueeze(1).repeat(1, features.size(1), 1) #hidden state expanded to match length of sequence
        energy = torch.tanh(self.attention(torch.cat((features, hidden_state), dim=2))) #compute energy score for each feauture 
        attention_weights = torch.nn.functional.softmax(self.v(energy).squeeze(2), dim=1) #softmax result to generate weights summing to 1, making it interpretable
        context = (features * attention_weights.unsqueeze(2)).sum(dim=1)
        return context, attention_weights

class CaptionDecoderWithAttention(torch.nn.Module): #updated the caption decoder to account for the attention mechanism
    def __init__(self, feature_dim, hidden_dim, vocab_size, embed_dim=512, num_layers=1):
        super(CaptionDecoderWithAttention, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(feature_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(embed_dim + feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions, hidden=None):
        batch_size = features.size(0)
        if hidden is None:
            hidden = (
                torch.zeros(1, batch_size, self.lstm.hidden_size).to(features.device),
                torch.zeros(1, batch_size, self.lstm.hidden_size).to(features.device),
            )

        embeddings = self.embedding(captions)
        outputs = []
        for t in range(embeddings.size(1)):
            context, _ = self.attention(features, hidden[0][-1])
            lstm_input = torch.cat((embeddings[:, t, :], context), dim=1).unsqueeze(1)
            output, hidden = self.lstm(lstm_input, hidden)
            outputs.append(self.fc(output.squeeze(1)))

        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden

def load_model():
    feature_extractor = torch.nn.Sequential(*list(models.vgg16(weights=models.VGG16_Weights.DEFAULT).children())[:-2])
    feature_extractor.eval()
    feature_extractor.to(device)

    decoder = CaptionDecoderWithAttention(
        feature_dim=512,
        hidden_dim=512,
        vocab_size=len(vocab),
        embed_dim=512,
        num_layers=1
    )
    decoder.load_state_dict(torch.load("models/decoder_attention.pth", map_location=device))
    decoder.eval()
    decoder.to(device)

    return feature_extractor, decoder

#changed generate caption to use beam seach - it gave a nonsensical result when using
def generate_caption_with_beam_search(image_path, feature_extractor, decoder, max_caption_length=20, beam_width=3, length_penalty=0.7, repetition_penalty=1.2, temperature=1.0):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(image_tensor)
        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)

    start_token = vocab['<SOS>']
    end_token = vocab['<EOS>']

    sequences = [[start_token]]
    scores = [0]

    for step in range(max_caption_length):
        all_candidates = []
        for i, seq in enumerate(sequences):
            if seq[-1] == end_token:
                all_candidates.append((seq, scores[i]))
                continue

            input_seq = torch.tensor([seq]).to(device)
            with torch.no_grad():
                logits, _ = decoder(features, input_seq)
                logits = logits[:, -1, :]
                probs = softmax(logits / temperature, dim=-1)

            top_probs, top_indices = torch.topk(probs, beam_width)

            for prob, idx in zip(top_probs[0], top_indices[0]):
                candidate = seq + [idx.item()]
                candidate_score = scores[i] - torch.log(prob).item()

                normalized_score = candidate_score / len(candidate)**length_penalty
                unique_tokens = len(set(candidate)) / len(candidate)
                total_score = normalized_score * unique_tokens**repetition_penalty

                all_candidates.append((candidate, total_score))

        all_candidates = sorted(all_candidates, key=lambda x: x[1])
        sequences, scores = zip(*all_candidates[:beam_width])

    best_sequence = sequences[0]
    return " ".join([idx_to_word[token] for token in best_sequence if token not in (start_token, end_token, vocab['<PAD>'])])

if __name__ == "__main__":
    feature_extractor, decoder = load_model()

    image_path = "untrained_images/motorbike.jpg"

    caption = generate_caption_with_beam_search(image_path, feature_extractor, decoder, beam_width=5)
    print("Generated Caption:", caption)
