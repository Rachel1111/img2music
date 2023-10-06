import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataProcess.getImgTextData import TensorDataProcess

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 构建MLP映射模型 初步模型，可能之后还要变
class ImageToTextMapping(nn.Module):
    def __init__(self, img_feature_dim, text_feature_dim):
        super(ImageToTextMapping, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(img_feature_dim, 1024),  # mapping to higher dimension
            nn.ReLU(),
            nn.Linear(1024, text_feature_dim)  # Mapping to the target dimension
        )

    def forward(self, img_features):
        img_features = img_features.to(torch.float32)  # transform the data type
        mapped_features = self.mapping(img_features)
        return mapped_features


if __name__ == "__main__":
    # 准备数据
    img_dir = r'../data/image'
    txt_dir = r'../data/text'
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100

    getdataset = TensorDataProcess(img_dir, txt_dir)
    # create Dataset and DataLoader
    dataset = getdataset.getTensorFeatures()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initial GPT-2 and mapping
    model_name = 'gpt2'
    img_feature_dim = 512  # 输入图像特征的维度
    text_feature_dim = 512  # 输出文本特征的维度


    model = ImageToTextMapping(img_feature_dim, text_feature_dim)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    for epoch in range(num_epochs):
        for img_features, text_targets in dataloader:
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            # forward
            predicted_text_features = model(img_features)

            # loss
            loss = criterion(predicted_text_features, text_targets)

            # backward and optimize
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'image_to_text_model_{epoch}.pth')
