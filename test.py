import torch
import torch.nn as nn
import time
import numpy as np
import os
import heapq

from utils import setup_logger, compute_metrics, relativeError

def test_model(model, test_loader, criterion, device, results_path='./'):
    """    
    Defines the model testing function applicable to both classification and regression problems.
    For classification tasks, the function returns the top 5 correctly classified samples with the highest probabilities for each class, for Grad-CAM analysis.
    For regression tasks, it returns the top 5 samples with the lowest absolute values of relative errors, for both upward and downward cases, also for Grad-CAM usage.
    """    
    
    since = time.time()
    
    model.eval()
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    logger = setup_logger(results_path, os.path.join(results_path,'loss.txt')) 
    
    test_total_loss, test_total_samples = 0, 0
    
    all_true = np.array([])
    all_pred = np.array([])
    all_score = np.array([])
    
    high_scores_0 = []
    high_scores_1 = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(torch.squeeze(output), target)
            test_total_loss += loss.item() * target.size(0)
            test_total_samples += target.size(0)
            
            all_true = np.concatenate([all_true, target.cpu().numpy()])
            if not isinstance(criterion, nn.MSELoss):
                scores = torch.softmax(output, dim=1)
                score, pred = torch.max(scores, dim=1)
                all_pred = np.concatenate([all_pred, pred.cpu().numpy()])
                all_score = np.concatenate([all_score, score.cpu().numpy()])
            else:
                pred = torch.squeeze(output)
                all_pred = np.concatenate([all_pred, pred.cpu().numpy()])
                scores = - np.abs(relativeError(pred.cpu(), target.cpu()))

            if not isinstance(criterion, nn.MSELoss):
                correct = pred == target
                correct_scores = scores[correct]
                correct_classes = target[correct]
                correct_imgs = data[correct]
            
                for score, img, clazz in zip(correct_scores, correct_imgs, correct_classes):
                    if clazz == 0:
                        if len(high_scores_0) < 5:
                            heapq.heappush(high_scores_0, (score[0].item(), img))
                        else:
                            heapq.heappushpop(high_scores_0, (score[0].item(), img))
                    elif clazz == 1:
                        if len(high_scores_1) < 5:
                            heapq.heappush(high_scores_1, (score[1].item(), img))
                        else:
                            heapq.heappushpop(high_scores_1, (score[1].item(), img))
            else:
                correct = scores > -1.0
                correct_scores = scores[correct]
                correct_target = target[correct] 
                correct_imgs = data[correct]
                for score, img, true_value in zip(correct_scores, correct_imgs, correct_target):
                    if true_value < 0:
                        if len(high_scores_0) < 5:
                            heapq.heappush(high_scores_0, (score, img))
                        else:
                            heapq.heappushpop(high_scores_0, (score, img))
                    elif true_value > 0:
                        if len(high_scores_1) < 5:
                            heapq.heappush(high_scores_1, (score, img))
                        else:
                            heapq.heappushpop(high_scores_1, (score, img))

    high_scores_0 = sorted(high_scores_0, reverse=True, key=lambda x: x[0])
    high_scores_1 = sorted(high_scores_1, reverse=True, key=lambda x: x[0])
    
    test_loss = test_total_loss / test_total_samples
    logger.info('{}'.format(test_loss))
    print(f"test_loss: {test_loss:.4f}")
    
    if isinstance(criterion, nn.MSELoss):
        metrics = compute_metrics(all_true, all_pred)
    else:
        metrics = compute_metrics(all_true, all_pred, all_score)
    with open(os.path.join(results_path,'metrics.txt'), "w") as f:
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, np.ndarray):
                f.write(f"{metric_name}:\n{metric_value}\n\n")
            else:
                f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    time_elapsed = time.time() - since
    print(f'Test complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    return high_scores_0, high_scores_1
