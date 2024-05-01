import os
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score, mean_absolute_error, r2_score

def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logger.setLevel(level)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

'''evaluation metrics'''
def correct_num(outputs, labels):
    with torch.no_grad():
        _, predicted = torch.max(outputs, dim=1)
        correct = (predicted == labels).sum().item()
        return correct

def relativeError(pred_val, true_val):
    pred_val = np.array(pred_val)
    true_val = np.array(true_val)
      
    both_zeros = (pred_val == 0) & (true_val == 0)
    
    relative_error = np.zeros_like(pred_val, dtype=float)
    
    non_zero_indices = ~both_zeros
    relative_error[non_zero_indices] = 2 * (pred_val[non_zero_indices] - true_val[non_zero_indices]) / (np.abs(pred_val[non_zero_indices]) + np.abs(true_val[non_zero_indices]))
    
    return relative_error

def sharpe_ratio(returns, risk_free_rate=0):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        return 0
    return (mean_return - risk_free_rate) / std_return

def compute_metrics(y_true, y_pred, y_scores=None):
    if y_scores is not None:
        metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_scores),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'pearson_corr': np.corrcoef(y_true, y_pred)[0, 1],
        'spearman_corr': np.corrcoef(np.argsort(y_true), np.argsort(y_pred))[0, 1],
        'sharpe_ratio': sharpe_ratio(y_pred)
        }
    else:
        metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(relativeError(y_pred, y_true)) * 100,
        'R_squared': r2_score(y_true, y_pred),
        'pearson_corr': np.corrcoef(y_true, y_pred)[0, 1],
        'spearman_corr': np.corrcoef(np.argsort(y_true), np.argsort(y_pred))[0, 1],
        'sharpe_ratio': sharpe_ratio(y_pred)
        }
    return metrics

'''plot'''
def plot_training_validation_loss(working_dir):
    """
    Plot training and validation loss from log file stored in the working directory.
    """
    data_path = os.path.join(working_dir, "losses.log")
    df = pd.read_csv(data_path, header=None)
    
    epochs = df.iloc[:, 0].to_numpy()
    train_loss = df.iloc[:, 1].to_numpy()
    valid_loss = df.iloc[:, 3].to_numpy()
    
    plot_path = os.path.join(working_dir, "training_validation_loss.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(epochs, train_loss, label='Training Loss')
    ax.plot(epochs, valid_loss, label='Validation Loss')

    min_pos = valid_loss.argmin() + 1
    ax.axvline(x=min_pos, linestyle='--', color='r', label='Early Stopping Checkpoint')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_xlim(1, len(train_loss) + 1)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)