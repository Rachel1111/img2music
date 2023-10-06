import os, torch, re
from torchtext.vocab import Vocab, build_vocab_from_iterator


def preprocess_text(text):
    # Use regular expression matching to keep only English words and punctuation marks and remove other characters
    cleaned_text = re.sub(r"[^a-zA-Z\s.,!?]", " ", text)
    return cleaned_text

def getVocab(dir):
    all_words = []
    special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']  # define special tokens
    all_words.extend(special_tokens)  # extend special tokens to vocab list
    txt_paths = [os.path.join(dir, filename) for filename in os.listdir(dir)]

    for path in txt_paths:
        sentences = open(path, 'r', encoding='utf-8').read()
        # text = preprocess_text(sentences)
        words = sentences.split()
        all_words.extend(words)

    # use build_vocab_from_iterator to create vocab
    vocab = build_vocab_from_iterator([all_words], specials=special_tokens)
    return vocab


def save_vocab(vocab, save_path):
    torch.save(vocab, save_path)

def load_vocab(load_path):
    vocab = torch.load(load_path)
    return vocab


if __name__=='__main__':
    dir = r'../data/text'
    vocab_path = '../data/vocab.pt'
    # check if the vocab exists
    if os.path.exists(vocab_path):
        # if it exists, load vocab
        vocab = load_vocab(vocab_path)
        print("Loaded existing vocabulary.")
    else:
        # if no, create vocab
        vocab = getVocab(dir)
        save_vocab(vocab, save_path=vocab_path)
        print("Built and saved vocabulary.")
    print(vocab.get_itos())


