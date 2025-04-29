import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def load_data(path, batch_size=100):
    # 讀取 CSV
    df = pd.read_csv(path, dtype=np.float32)
    
    # 分離 features 和 labels
    features_numpy = df.loc[:, df.columns != 'label'].to_numpy() / 255
    targets_numpy = df['label'].to_numpy()

    # 分割訓練與驗證集
    features_train, features_val, targets_train, targets_val = train_test_split(
        features_numpy, targets_numpy, test_size=0.2, random_state=0
    )

    # numpy → tensor
    featurestrain = torch.from_numpy(features_train)
    targetstrain = torch.from_numpy(targets_train).type(torch.LongTensor)
    featuresval = torch.from_numpy(features_val)
    targetsval = torch.from_numpy(targets_val).type(torch.LongTensor)

    # 建立 DataLoader
    train_ds = TensorDataset(featurestrain, targetstrain)
    val_ds = TensorDataset(featuresval, targetsval)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, features_numpy, targets_numpy