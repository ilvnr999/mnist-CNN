import matplotlib.pyplot as plt

def show_sample(features_numpy, targets_numpy, idx=10):
    plt.imshow(features_numpy[idx].reshape(28, 28))
    plt.title(str(targets_numpy[idx]))
    plt.show()

def plot_metrics(train_loss, val_loss, train_acc, val_acc, epochs):
    plt.plot(range(epochs), train_loss, label='training loss', color='blue')
    plt.plot(range(epochs), val_loss, label='validation loss', color='red')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.show()

    plt.plot(range(epochs), train_acc, label='training accuracy', color='blue')
    plt.plot(range(epochs), val_acc, label='validation accuracy', color='red')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy Curve")
    plt.show()