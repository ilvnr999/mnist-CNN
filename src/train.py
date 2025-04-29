import torch
import torch.nn as nn

def train_model(model, train_loader, val_loader, device, epochs=30, learning_rate=0.1):
    error = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 可選：記錄訓練過程
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    count = 0
    
    for epoch in range(epochs):
        # 訓練階段
        model.train()
        total = 0
        correct = 0
        for images, labels in train_loader:
            X = images.reshape(-1, 1, 28, 28).to(device)
            y = labels.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = error(outputs, y)
            loss.backward()
            optimizer.step()

            total += len(y)
            _, predicted = torch.max(outputs.data, axis=1)
            correct += (predicted == y).sum().item()

            count += 1
            if count % 336 == 0:
                acc = (100 * correct / total)
                train_loss_list.append(loss.item())
                train_acc_list.append(acc)

        # 驗證階段
        model.eval()
        val_loss = 0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.reshape(-1, 1, 28, 28).to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_loss += error(outputs, labels).item()
                _, predicted = torch.max(outputs.data, axis=1)
                val_correct += (predicted == labels).sum().item()
                val_total += len(labels)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / val_total
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)
        print(f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%")

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list