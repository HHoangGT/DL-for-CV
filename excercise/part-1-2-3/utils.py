import os
import json
import matplotlib.pyplot as plt


def load_config(config_path="config.json"):
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def save_results(results_dict, filepath="result/results.json"):
    """
    Saves or appends results to a JSON file.
    results_dict: dictionary containing training records
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        try:
            with open(filepath, encoding="utf-8") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = {}
    else:
        existing_data = {}

    existing_data.update(results_dict)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4)
    print(f"Results saved/updated successfully to {filepath}")


def plot_history(model_name, history, save_dir="images"):
    """
    Plots the training and testing loss, and testing accuracy over epochs.
    Saves the figures as PNG images.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.plot(epochs, history["test_loss"], "r-", label="Test (Val) Loss")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["test_accuracy"], "g-", label="Test Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, f"{model_name}_history.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")
