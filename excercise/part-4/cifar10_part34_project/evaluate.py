import argparse

import torch
from torch import nn

from src.data import create_dataloaders
from train import build_model
from src.utils import get_device, load_config
from src.engine import evaluate



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config.get("device", "cuda"))

    _, _, test_loader = create_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 2),
        val_split=config.get("val_split", 0.1),
        seed=config.get("seed", 42),
    )

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device)
    print({"test_loss": metrics["loss"], "test_acc": metrics["acc"]})


if __name__ == "__main__":
    main()
