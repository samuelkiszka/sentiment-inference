import time
import matplotlib.pyplot as plt
import os

from src.data_loader import DataLoader
import src.transformer_encoder as te
import src.trainer as tr

def train_model(num_per_class, epochs=50, batch_size=32, lr=0.001, note=None):
    os.makedirs(f"results/sample_{num_per_class}", exist_ok=True)
    note_str = ", " + note if note else ""
    result_img_path = f"results/sample_{num_per_class}/sample_{num_per_class}_batch_{batch_size}_lr_{lr}{note_str}.png"
    if os.path.exists(result_img_path):
        print(f"Results for sample size {num_per_class} and batch size {batch_size} and lr {lr} already exist. Skipping training.")
        return None, None, None, None

    train_dataloader = DataLoader(split="train", batch_size=batch_size, num_per_class=num_per_class)
    test_dataloader = DataLoader(split="test", batch_size=batch_size, num_per_class=num_per_class)

    model = te.TransformerClassifier(d_model=768, num_layers=1, n_heads=1, d_ff=64, num_classes=2)
    trainer = tr.Trainer(model, train_dataloader, test_dataloader, lr=lr)

    train_stats = []
    test_stats = []
    best_acc = 0

    print(f"Starting training for {epochs} epochs with {batch_size} batch size with LR {lr} on {num_per_class * 2} samples...")
    times = [time.time()]
    for epoch in range(1, epochs + 1):
        print(f"--- Epoch {epoch} ---")
        train_loss, train_acc = trainer.train_epoch()
        test_loss, test_acc = trainer.evaluate()

        saved = False
        if test_acc > best_acc:
            best_acc = test_acc
            save_dir = f"models/sample_{num_per_class}"
            os.makedirs(save_dir, exist_ok=True)
            model.save(f"{save_dir}/best_model.npz")
            saved = True

        train_stats.append((train_loss, train_acc))
        test_stats.append((test_loss, test_acc))
        print(f"Time taken for epoch {epoch}: {time.time() - times[-1]:.2f} seconds")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        if saved:
            print(f"New best model saved")
        times.append(time.time())
    time_finish = time.time() - times[0]
    print(f"Finished training in {time_finish:.2f} seconds")

    return True, train_stats, test_stats, time_finish

def plot_results(sample_size, batch_size, train_stats, test_stats, time_sec, lr=0.001, note=None):
    time_min = round(time_sec / 60, 2)
    epochs = list(range(1, len(train_stats) + 1))
    train_losses, train_accs = zip(*train_stats)
    test_losses, test_accs = zip(*test_stats)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim([0, 1])
    plt.title(f'Training and Test Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.title(f'Training and Test Accuracy')
    plt.legend()

    note_str = ", Note: " + note if note else ""

    plt.suptitle(f"Sample: {sample_size}, Batch: {batch_size}, LR:{lr}, Time: {time_min:.2f} min{note_str}")
    plt.tight_layout()

    os.makedirs(f"results/sample_{sample_size}", exist_ok=True)
    plt.savefig(f"results/sample_{sample_size}/sample_{sample_size}_batch_{batch_size}_lr_{lr}{note_str}.png")

if __name__ == "__main__":
    epochs = 5
    num_per_class = 250
    batch_size = 2
    plot, train, test, total_time = train_model(num_per_class, epochs, batch_size)
    if plot is not None:
        plot_results(num_per_class, batch_size, train, test, total_time)
    print("Training complete.")
