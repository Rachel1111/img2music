import clip, torch
from torchtext.vocab import Vocab
from collections import Counter
from torch.utils.data import Dataset
from dataProcess.process import clip_model

class TextFeatures(Dataset):
    def __init__(self, txt_paths, vocab):
        self.txt_paths = txt_paths
        self.vocab = vocab
        self.max_length = 77

    def __len__(self):
        return len(self.txt_paths)

    def __getitem__(self, index):
        txt_path = self.txt_paths[index]

        text = open(txt_path, 'r', encoding='utf-8').read()
        tokens = clip.tokenize(text[:self.max_length])  # use clip.tokenize to transform text to tokens
        text_feature = clip_model.encode_text(tokens)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

        # transform text into index
        text_tokens = text.split()[:self.max_length-2]  # set max_length

        # add（<sos>）at the star, and add（<eos>）at the end
        text_indices = [self.vocab['<bos>']] + [self.vocab[token] for token in text_tokens] + [self.vocab['<eos>']]
        # Padding to ensure equal length
        if len(text_indices) < self.max_length:
            text_indices += [self.vocab['<pad>']] * (self.max_length - len(text_indices))

        text_indices = torch.tensor(text_indices, dtype=torch.long)

        return text_feature, text_indices