import torch, os, scipy
from PIL import Image
from module.image2Text import ImageToTextMapping
from module.textFeaure2Text import TextFeature2Text
from generatetext import generated
from dataProcess.getVocab import *
from dataProcess.process import clip_model, clip_preprocess, device
from transformers import AutoProcessor, MusicgenForConditionalGeneration

def createModels(model_path1, model_path2):
    # create  image2textMapping model
    img_feature_dim = 512
    text_feature_dim = 512
    mapping_model = ImageToTextMapping(img_feature_dim, text_feature_dim)
    # if the model trains on GPU, uses on CPU
    mapping_model.load_state_dict(torch.load(model_path1, map_location=torch.device('cpu')))
    mapping_model.eval()

    # create textfeature2text model
    input_dim = 512
    model_name = 'gpt2'
    # create model param
    text_model = TextFeature2Text(input_dim, model_name)
    # load
    text_model.load_state_dict(torch.load(model_path2))
    text_model.eval()

    # create text2music model
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    text2music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    return mapping_model, text_model, text2music_model, processor


# use ImageToTextMapping model
def image2text(model, image_path):
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    image_feature = clip_model.encode_image(image)
    text_feature = model(image_feature)
    return text_feature

# use Textfeature2Text model
def textfeature2text(model, text_feature):
    # prepare data
    max_length = 77
    temperature = 0.1
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

    generated_text = generated(model, text_feature)
    return generated_text

# use text2music model
def textGeneratedMusic(model, processor, text):
    inputs = processor(
        text=text,
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs, max_new_tokens=256)
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())


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

    image_path = 'img.jpg'
    model_path1 = 'models/image_to_text_mapping_model.pth'
    model_path2 = 'models/gpt2_generate.pth'
    mapping_model, text_model, music_model, processor = createModels(model_path1, model_path2)

    text_feature = image2text(mapping_model, image_path)
    generated_text = textfeature2text(model_path2, text_feature)
    textGeneratedMusic(music_model, processor, generated_text)
