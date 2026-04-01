import torch
from dataset import get_cifar10_loaders
from models import SoftmaxRegression, MLP, CNN, ViTPyTorch
from custom_vit import ViTCustom
from train import train_model
from utils import load_config, save_results, plot_history


def main():
    config = load_config()
    device = (
        config["training"]["device"]
        if torch.cuda.is_available() and config["training"]["device"] == "cuda"
        else "cpu"
    )
    print(f"Using device: {device}")

    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    patience = config["training"].get("early_stopping_patience", 5)
    batch_size = config["dataset"]["batch_size"]
    num_workers = config["dataset"]["num_workers"]

    print("Loading CIFAR-10 Dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, num_workers=num_workers
    )

    models_dict = {
        "softmax": SoftmaxRegression,
        "mlp": MLP,
        "cnn": CNN,
        "vit_pytorch": ViTPyTorch,
        "vit_custom": ViTCustom,
    }

    results = {}

    for model_name, model_class in models_dict.items():
        if config["models_to_run"].get(model_name, False):
            print(f"\n{'=' * 50}")
            print(f"Training Model: {model_name}")
            print(f"{'=' * 50}")

            model = model_class()

            res = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=epochs,
                lr=lr,
                patience=patience,
            )

            results[model_name] = res

            # Save metrics to JSON incrementally
            save_results({model_name: res})

            # Plot and save visualize history
            plot_history(model_name, res["history"], save_dir="images")

    print("\nAll enabled models finished training.")


if __name__ == "__main__":
    main()
