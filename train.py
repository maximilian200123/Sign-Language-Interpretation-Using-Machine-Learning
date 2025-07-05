import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from datetime import datetime
import numpy as np

from dataset_loader import LandmarksDataset, collate_fn
from architecture.RNN import HandSignClassifier

def train_model(
        model,
        train_loader,
        val_loader,
        num_classes,
        epochs=20,
        lr=0.001,
        checkpoint_dir='checkpoints',
        log_dir='runs',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        patience=10
):
    
    os.makedirs(checkpoint_dir,exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_dir,datetime.now().strftime("%Y%m%d-%H%M%S")))
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    best_loss_path = os.path.join(checkpoint_dir, f"2_layer_best_model_loss_{num_classes}_classes.pth")
    best_acc_path = os.path.join(checkpoint_dir, f"2_layer_best_model_acc_{num_classes}_classes.pth")

    print("Starting training...")

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x,lengths,y in train_loader:
            
            x,lengths,y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x,lengths)
            loss = criterion(outputs,y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        training_acc = 100 * correct/total
        avg_loss = running_loss / len(train_loader)
        
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for x_val, lengths_val, y_val in val_loader:
                x_val, lengths_val, y_val = x_val.to(device), lengths_val.to(device), y_val.to(device)
                outputs = model(x_val, lengths_val)
                loss = criterion(outputs,y_val)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_val).sum().item()
                val_total += y_val.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning Rate",current_lr,epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_loss_path) 
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} — no improvement in {patience} epochs.")
                break 

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),best_acc_path)

        writer.add_scalar("Loss/Train",avg_loss,epoch)
        writer.add_scalar("Loss/Val",val_loss,epoch)
        writer.add_scalar("Accuracy/Train",training_acc,epoch)
        writer.add_scalar("Accuracy/Val",val_acc,epoch) 

        print(f"[Epoch {epoch:02d}] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {training_acc:.2f}% | Val Acc: {val_acc:.2f}% | Current learning rate {current_lr:6f}" )

        checkpoint_path = os.path.join(checkpoint_dir,f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(),checkpoint_path)

    writer.close()


if __name__ == '__main__':

    batch_size = 16
    input_size = 126
    hidden_size = 256

    train_set = LandmarksDataset("wlasl100",mode = "train", max_glosses=100)
    val_set = LandmarksDataset("wlasl100",mode = "val", max_glosses=100)

    num_classes = len(set(train_set.label_to_idx))

    np.save("idx_to_label" +f"_{num_classes}_classes.npy", train_set.idx_to_label)

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
    
    model = HandSignClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes, dropout=0.3)

    train_model(model, train_loader, val_loader, num_classes,epochs=120)

