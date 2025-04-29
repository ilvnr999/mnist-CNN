import matplotlib.pyplot as plt

def show_sample(features_numpy, targets_numpy, idx=10):
    plt.imshow(features_numpy[idx].reshape(28, 28))
    plt.title(str(targets_numpy[idx]))
    plt.show()