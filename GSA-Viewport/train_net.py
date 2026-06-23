import numpy as np
import torch
import os
import time
from torch.utils.data import DataLoader
from datasets.dataset import CocoDataset
from model.model import TransformerWithGCN
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
annotation_file_train = r"dataset/annotations/instances_train2017.json"
annotation_file_val = r"dataset/annotations/instances_val2017.json"
image_root = r"dataset/images"
transformer_args = {
    "d_model": 512,
    "nhead": 16,
    "num_encoder_layers": 6,
    "num_decoder_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.2,
    "activation": "gelu",
    "normalize_before": False,
    "return_intermediate_dec": False
}
gcn_in_features = 512
gcn_out_features = 512
drop_rate = 0.2
batch_size = 16
epochs = 100

early_stop_patience = 10
best_val_acc = 0.0
early_stop_counter = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerWithGCN(transformer_args, gcn_in_features, gcn_out_features, drop_rate).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

weight = torch.tensor([2.0]).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

def save_model(model, optimizer, epoch, save_path="model_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for proposals, proposal_features, adj, labels, padding in dataloader:
            proposals, proposal_features, adj, labels, padding = (
                proposals.to(device),
                proposal_features.to(device),
                adj.to(device),
                labels.to(device),
                padding.to(device)
            )
            hs, _ = model(proposals, proposal_features, adj, labels, padding)
            labels = labels.float().unsqueeze(-1)
            loss = criterion(hs, labels)
            total_loss += loss.item()

            preds = (hs > 0).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = (all_preds == all_labels).sum() / len(all_labels)

    print(f"Validation - Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    model.train()
    return avg_loss, acc, precision, recall, f1

# 训练主循环
def train():
    global early_stop_counter
    best_f1 = 0.0
    train_dataset = CocoDataset(annotation_file_train, os.path.join(image_root, "train"))
    val_dataset = CocoDataset(annotation_file_val, os.path.join(image_root, "val"))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):

        model.train()
        epoch_loss = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_dataloader):
            proposals, proposal_features, adj, labels, padding = batch
            proposals, proposal_features, adj, labels, padding = (
                proposals.to(device),
                proposal_features.to(device),
                adj.to(device),
                labels.to(device),
                padding.to(device)
            )

            hs, _ = model(proposals, proposal_features, adj, labels, padding)
            labels = labels.float().unsqueeze(-1)

            loss = criterion(hs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item()
            print(
                f"Training Batch {batch_idx + 1}/{len(train_dataloader)} - Loss: {loss.item():.4f} ")
        elapsed_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}")

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_dataloader, criterion)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            early_stop_counter = 0
            save_model(model, optimizer, epoch, save_path=f"model_best.pth")
        else:
            early_stop_counter += 1
            print(f"No F1 improvement in {early_stop_counter} epochs")

        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered")
            break

        if epoch % 5 == 0:
            save_model(model, optimizer, epoch, save_path=f"model_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()
