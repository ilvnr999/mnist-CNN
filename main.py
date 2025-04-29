# main.py
import torch
from train import train_model
from models.cnn_model import CNNmodel
from utils import plot_metrics

# 設定 device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda is available.")
else:
    device = torch.device("cpu")
    print("using CPU.")

# 初始化模型
model = CNNmodel().to(device)

# 開始訓練
train_loss_list, val_loss_list, train_acc_list, val_acc_list = train_model(model, device)

# 畫圖
plot_metrics(train_loss_list, val_loss_list, train_acc_list, val_acc_list, epochs=30)