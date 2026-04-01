import torch
import torch.nn as nn
import torch.optim as optim
import time


def train_model(
    model, train_loader, test_loader, device, epochs=100, lr=0.001, patience=5
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "test_loss": [],
        "test_accuracy": [],
        "epoch_times": [],
    }

    start_train_time = time.time()
    best_test_loss = float("inf")
    epochs_no_improve = 0
    actual_epochs_run = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_time = time.time() - epoch_start

        # Evaluation step
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader.dataset)
        test_acc = correct / total

        history["train_loss"].append(epoch_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_acc)
        history["epoch_times"].append(epoch_time)
        actual_epochs_run += 1

        print(
            f"Epoch {epoch + 1}/{epochs} - Time: {epoch_time:.2f}s "
            f"- Train Loss: {epoch_loss:.4f} "
            f"- Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}"
        )

        # Early Stopping check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
            # Optional: save the best model weights here
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs with no improvement in test loss."
                )
                break

    total_time = time.time() - start_train_time
    print(f"Total training time: {total_time:.2f}s")

    return {
        "total_time_seconds": total_time,
        "final_test_accuracy": history["test_accuracy"][-1],
        "final_train_loss": history["train_loss"][-1],
        "history": history,
        "num_epochs": actual_epochs_run,
    }
