from data import open_buckets
from training import evaluate, init, save_checkpoint, train_one_epoch


def main() -> None:
    checkpoint_dir = "checkpoints"
    device, model, optimizer, criterion, writer, start_epoch = init(
        checkpoint_dir, resume_from_checkpoint=True
    )

    total_epochs = 10

    with open_buckets("../4_tokens/train") as trainset, open_buckets("../4_tokens/val") as valset:
        for epoch in range(start_epoch, total_epochs + 1):
            # Train for one epoch
            train_one_epoch(device, trainset, model, optimizer, criterion, writer, epoch)

            # Evaluate and get validation loss
            val_loss = evaluate(device, valset, model, criterion, writer, epoch)

            # Save checkpoint after each epoch (includes model weights)
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir)

    # Close TensorBoard writer
    writer.close()  # type: ignore


if __name__ == "__main__":
    main()
