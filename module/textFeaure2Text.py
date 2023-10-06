import torch, os
import torch.optim as optim
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataProcess.getVocab import *
from dataProcess.getTextFeature import TextFeatures
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class TextFeature2Text(nn.Module):
    def __init__(self, clip_feature_dim=512, model_name='gpt2'):
        super(TextFeature2Text, self).__init__()
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.n_embd = 768

        # The incoming clip text features have to be changed to the input dimension of gpt2
        self.text_feature_to_gpt2 = nn.Linear(clip_feature_dim, self.n_embd)

    def forward(self, clip_feature, text_indices):
        # Transformer.word_embedding word embedding for gpt2 with word embedding dimension 768
        text_len = len(text_indices)
        clip_embed = self.gpt2_model.transformer.wte(text_indices)
        clip_feature_gpt2 = self.text_feature_to_gpt2(clip_feature)
        # map_clip_feature = clip_feature_gpt2.view(-1, self.clip_len, self.n_embd)
        # print(map_clip_feature.shape)
        embedding_cat = torch.cat((clip_feature_gpt2, clip_embed), dim=1)
        out = self.gpt2_model(inputs_embeds=embedding_cat)
        # logits:[bs, clip_embed_len+map_clip_feature_len, n_embd]
        logits = out.logits
        return logits

def train():
    # 准备数据
    txt_dir = r'../data/text'
    txt_paths = [os.path.join(txt_dir, filename) for filename in os.listdir(txt_dir)]
    vocab_path = '../data/vocab.pt'
    # check if the vicab exists
    if os.path.exists(vocab_path):
        # if yes, load vocab
        vocab = load_vocab(vocab_path)
        print("Loaded existing vocabulary.")
    else:
        # if no, create vocab
        vocab = getVocab(txt_dir)
        save_vocab(vocab, save_path=vocab_path)
        print("Built and saved vocabulary.")

    dataset = TextFeatures(txt_paths, vocab)
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 100
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # create model and optimizer
    input_dim = 512  # input feature dimension
    output_dim = len(vocab)
    model_name = 'gpt2'
    model = TextFeature2Text(input_dim, model_name)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # train
    for epoch in range(num_epochs):
        for text_features, text_indices in dataloader:
            optimizer.zero_grad()

            text_indices.to(device)
            # input: text feature and text index
            logits = model(text_features, text_indices)
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            target_indices = text_indices.view(-1)

            loss = criterion(shift_logits, target_indices)

            # backward and optimize
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f'gpt2_generate_{epoch}.pth')


if __name__ == '__main__':
    train()