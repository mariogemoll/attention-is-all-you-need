import torch

from data import open_buckets
from training import evaluate, init, train_one_epoch


def main() -> None:
    device, model, optimizer, criterion, writer = init()

    with open_buckets("../4_tokens/train") as trainset, open_buckets("../4_tokens/val") as valset:
        for epoch in range(1, 11):
            train_one_epoch(device, trainset, model, optimizer, criterion, writer, epoch)
            model.load_state_dict(torch.load(f"model_epoch_{epoch}.pt", map_location=device))
            evaluate(device, valset, model, criterion, writer, epoch)
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")

    # Close TensorBoard writer
    writer.close()  # type: ignore


if __name__ == "__main__":
    main()
