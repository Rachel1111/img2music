import clip
from PIL import Image
import torch
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load clip model
clip_model, clip_preprocess = clip.load('ViT-B/32', device)

class DataProcess(Dataset):

    def __init__(self, img_paths, txt_paths):
        self.img_paths = img_paths
        self.txt_paths = txt_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        txt_path = self.txt_paths[index]

        # image = Image.open(img_path)
        # text = open(txt_path, 'r', encoding='utf-8').read()
        image = Image.open(img_path)
        image = clip_preprocess(image).unsqueeze(0).to(device)  # transform to tensor and move to device
        text = open(txt_path, 'r', encoding='utf-8').read()
        text = clip.tokenize(text[:50]).to(device)  # use clip.tokenize to transform text to tokens

        return image, text

class GetFeatures(object):

    def __init__(self):
        self.max_length = 50

    # CLIP uses a shared visual-text encoder for both image and text encoding,
    # which results in consistent dimensionality of the features generated [1, 512]
    def imageAndText(self, image, text):
        # 对图像编码
        # image = clip_preprocess(image).unsqueeze(0).to(device)
        image_feature = clip_model.encode_image(image)

        # The maximum length of CLIP text is 77, if text is encoded and truncation is required.
        # text = clip.tokenize(txt[:self.max_length]).to(device)
        text_feature = clip_model.encode_text(text)

        # L2 normalisation, calculate the similarity,
        # and if the similarity is too low, don't want the pair of graphic pairs
        # Symbols like /= cause pytorch to operate in place and need to be changed to the full operator
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        similarity = text_feature.detach().cpu().numpy() @ image_feature.detach().cpu().numpy().T

        return image_feature, text_feature, similarity
