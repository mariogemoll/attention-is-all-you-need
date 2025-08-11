import torch

from data import open_buckets
from training import evaluate, init, train_one_epoch


def main() -> None:
    device, model, optimizer, criterion = init()

    with open_buckets("../4_tokens/train") as trainset, open_buckets("../4_tokens/val") as valset:
        for epoch in range(1, 11):
            train_one_epoch(device, trainset, model, optimizer, criterion, epoch)
            model.load_state_dict(torch.load(f"model_epoch_{epoch}.pt", map_location=device))
            evaluate(device, valset, model, criterion, epoch)
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
