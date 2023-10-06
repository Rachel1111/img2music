import clip
from dataProcess.getVocab import *
from module.textFeaure2Text import TextFeature2Text
from dataProcess.process import clip_model

import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """use Top-k and Top-p filter logits for better generated text.

    Args:
        logits (torch.Tensor): generated logits by model，shape [batch_size, vocab_size]。
        top_k (int): Number of tokens with the highest probability to be retained。
        top_p (float): Threshold for the cumulative probability to be retained.
        filter_value (float): need filtered value

    Returns:
        torch.Tensor: filtered logits，shape [batch_size, vocab_size]。
    """
    if top_k > 0:
        # Set all but top_k highest probability tokens to -inf
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # except tokens that probability less than top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        for i in range(logits.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def generated(model, clip_feature):
    # define generated token
    generated_indices = []

    input_embed = model.text_feature_to_gpt2(clip_feature)
    input_embed = input_embed.to(torch.int64)

    for i in range(max_length):
        logits = model.gpt2_model(input_embed)
        predicted_logits = logits.logits

        # Taking the predictive distribution of the last word
        next_token_logits = predicted_logits[..., -1, :]
        next_token_logits = next_token_logits / temperature
        filter_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=0.5)
        next_token_id = torch.multinomial(F.softmax(filter_logits, dim=-1), num_samples=1).squeeze(1).tolist()
        # print(next_token_ids)

        if next_token_id in [0, 1, 2, 3]:
            break

        generated_indices.append(next_token_id)
        next_token_id = torch.tensor(next_token_id)
        next_token_embeds = model.gpt2_model.transformer.wte(next_token_id).unsqueeze(1)
        inputs_embeds = torch.cat((input_embed, next_token_embeds), dim=1)

    # generated_indices
    generated_text = ' '.join([vocab[idx.item()] for idx in generated_indices])
    return generated_text


if __name__=='__main__':
    txt_dir = r'data/text'
    txt_paths = [os.path.join(txt_dir, filename) for filename in os.listdir(txt_dir)]
    vocab_path = 'data/vocab.pt'
    if os.path.exists(vocab_path):
        # 如果文件存在，加载词汇表
        vocab = load_vocab(vocab_path)
        print("Loaded existing vocabulary.")
    else:
        # 如果文件不存在，构建并保存词汇表
        vocab = getVocab(txt_dir)
        save_vocab(vocab, save_path=vocab_path)
        print("Built and saved vocabulary.")

    input_dim = 512
    model_name = 'gpt2'
    # create model
    model = TextFeature2Text(input_dim, model_name)
    # load model param
    model.load_state_dict(torch.load('models/gpt2_generate_64.pth'))
    # set model as evaluation
    model.eval()


    # prepare data
    max_length = 77
    temperature = 0.1
    text = 'a cat is sitting on a sofa'
    tokens = clip.tokenize(text[:max_length])  # use clip.tokenize to transform text to tokens
    text_feature = clip_model.encode_text(tokens)
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

    generated_text = generated(model, text_feature)
