import time
import matplotlib.pyplot as plt
import os

from src.data_loader import DataLoader, DATA_MODEL, N_CLASSES
import src.transformer_encoder as te
import src.trainer as tr


def train_model(dataset_size=250, n_layers=1, n_heads=1, d_ff=64,
                epochs=50, batch_size=32, lr=2e-5, optimizer="sgd", note=None, save_plot=False, save_model=True):
    os.makedirs(f"results/sample_{dataset_size}", exist_ok=True)
    note_str = f"_{note}" if note else ""
    result_img_path = f"results/sample_{dataset_size}/sample_{dataset_size}_batch_{batch_size}_lr_{lr}{note_str}.png"
    if os.path.exists(result_img_path):
        print(f"Results for sample size {dataset_size} and batch size {batch_size} and lr {lr} already exist. Skipping training.")

    train_dataloader = DataLoader(split="train", batch_size=batch_size, num_per_class=dataset_size)
    test_dataloader = DataLoader(split="test", batch_size=batch_size, num_per_class=dataset_size)

    model = te.TransformerClassifier(d_model=DATA_MODEL, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, n_classes=N_CLASSES)
    trainer = tr.Trainer(model, train_dataloader, test_dataloader, optimizer, lr=lr)

    train_stats = []
    test_stats = []
    best_acc = 0

    print(f"Starting training for {epochs} epochs with {batch_size} batch size with LR {lr} on {dataset_size * 2} samples...")
    times = [time.time()]
    for epoch in range(1, epochs + 1):
        print(f"--- Epoch {epoch} ---")
        train_loss, train_acc = trainer.train_epoch()
        test_loss, test_acc = trainer.evaluate()

        print(f"Time taken for epoch {epoch}: {time.time() - times[-1]:.2f} seconds")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        if save_model and test_acc > best_acc:
            best_acc = test_acc
            save_dir = f"models/sample_{dataset_size}"
            os.makedirs(save_dir, exist_ok=True)
            model.save(f"{save_dir}/best_model{note}.npz")
            print(f"New best model saved")

        train_stats.append((train_loss, train_acc))
        test_stats.append((test_loss, test_acc))
        times.append(time.time())

    time_sec = time.time() - times[0]
    print(f"Finished training in {time_sec:.2f} seconds")

    if save_plot:
        print(f"Saving training plot to {result_img_path} ...")
        plot_results(dataset_size, batch_size, train_stats, test_stats, time_sec, lr=lr, note=note)

def plot_results(sample_size, batch_size, train_stats, test_stats, time_sec, lr=2e-5, note=None):
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

    note_str = f"_{note}" if note else ""
    os.makedirs(f"results/sample_{sample_size}", exist_ok=True)
    plt.savefig(f"results/sample_{sample_size}/sample_{sample_size}_batch_{batch_size}_lr_{lr}{note_str}.png")

if __name__ == "__main__":
    NUM_PER_CLASS = 250             # 500 total samples for training (250 positive, 250 negative) (default: 250)
    N_LAYERS =      2               # number of transformer encoder layers (default: 2)
    N_HEADS =       4               # number of attention heads per layer (default: 4)
    D_FF =          64              # Feedforward network dimension (default: 64)
    EPOCHS =        20              # number of training epochs (default: 20)
    BATCH_SIZE =    32              # Batch size (default: 32)
    LR =            2e-5            # Learning rate (default: 2e-5)
    OPTIM =         "adamw"         # Optimizer: 'adamw' or 'sgd'
    SAVE_PLOT =     True            # Save training plot
    SAVE_MODEL =    True            # Save best model
    NOTE =          "l2h4bff6432"   # number of layers, heads, batch size note for saving files (default: "l2h4ff64b32")

    train_model(dataset_size=NUM_PER_CLASS,
                n_layers=N_LAYERS,
                n_heads=N_HEADS,
                d_ff=D_FF,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
                optimizer=OPTIM,
                note=NOTE,
                save_plot=SAVE_PLOT,
                save_model=SAVE_MODEL)
