import torch
import time
import csv
import os

from utils import correct_num, setup_logger


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    
    def __init__(self, patience=7, delta=1e-4, path='checkpoint.pt',verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, epoch, in_cooldown, model=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if model is not None:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score * (1. -self.delta):
            if not in_cooldown:
                self.counter += 1
                print(f'EarlyStopping counter(based on validation): {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            if model is not None:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        with open(self.path.replace('.pth', '.csv'), 'w', newline='') as csvfile:
            fieldnames = ['best_epoch', 'best_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'best_epoch': self.best_epoch, 'best_loss': -self.best_score})

def train_model(model, train_loader, valid_loader, optimizer, criterion, device, n_epochs, patience, results_path='./'):
    """
    Defines the model training function that employs early stopping to prevent overfitting.
    Suitable for both classification and regression problems
    """
    
    since = time.time()
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    logger_losses = setup_logger(results_path, os.path.join(results_path, 'losses.log'))
    
    early_stopping = EarlyStopping(patience=patience, path=os.path.join(results_path, 'model_earlystopping.pth'))

    
    for epoch in range(1, n_epochs + 1):
        train_total_loss, train_total_correct, train_total_samples = 0, 0, 0
        valid_total_loss, valid_total_correct, valid_total_samples = 0, 0, 0
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(torch.squeeze(output), target)
            loss.backward()
            optimizer.step()
            correct = correct_num(output, target)
            
            for param in model.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print("Error: model parameters contain NaN or inf.")
                    
            train_total_loss += loss.item() * target.size(0)
            train_total_correct += correct
            train_total_samples += target.size(0)

        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(torch.squeeze(output), target)
                correct = correct_num(output, target)
                valid_total_loss += loss.item() * target.size(0)
                valid_total_correct += correct
                valid_total_samples += target.size(0)

        train_loss, valid_loss = train_total_loss / train_total_samples, valid_total_loss / valid_total_samples
        train_acc, valid_acc = train_total_correct / train_total_samples, valid_total_correct / valid_total_samples
        logger_losses.info('{},{},{},{},{}'.format(
            epoch,
            train_loss,
            train_acc,
            valid_loss,
            valid_acc
            )
                           )
        print(f"[{epoch}/{n_epochs}] train_loss: {train_loss:.4f} train_accuracy: {train_acc:.4f}; valid_loss: {valid_loss:.4f}  valid_accuracy: {valid_acc:.4f}")
        
        train_total_loss, train_total_correct, train_total_samples = 0, 0, 0
        valid_total_loss, valid_total_correct, valid_total_samples = 0, 0, 0
        
        early_stopping(valid_loss, epoch, False, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    model.load_state_dict(torch.load(os.path.join(results_path, 'model_earlystopping.pth')))

    return model
