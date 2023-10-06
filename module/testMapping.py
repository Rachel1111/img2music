import torch
from PIL import Image
from module.image2Text import ImageToTextMapping
from dataProcess.process import clip_model, clip_preprocess, device
from torch.utils.data import DataLoader
from dataProcess.getImgTextData import TensorDataProcess

def calculate_accuracy(model, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # Forward pass to get predictions
            outputs = model(images)
            # Get predicted labels
            _, predicted = torch.max(outputs.data, 1)
            # Count correct predictions
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':

    img_feature_dim = 512
    text_feature_dim = 512
    mapping_model = ImageToTextMapping(img_feature_dim, text_feature_dim)
    mapping_model.to(device)

    # load model
    model_path = '../model/image_to_text_mapping_model.pth'
    mapping_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    img_dir = 'data/image'
    txt_dir = 'data/text'

    getdataset = TensorDataProcess(img_dir, txt_dir)
    # create Dataset and DataLoader
    dataset = getdataset.getTensorFeatures()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    accuracy = calculate_accuracy(mapping_model, dataloader)
    print(accuracy)