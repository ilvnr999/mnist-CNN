import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_and_preprocess_data(train_df, batch_size=100):
    features_numpy = train_df.loc[:, train_df.columns != 'label'].to_numpy() / 255
    targets_numpy = train_df['label'].to_numpy()

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

    return train_loader, val_loader