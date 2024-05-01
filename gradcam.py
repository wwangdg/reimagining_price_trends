import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import csv

def runGradCam(model, images_scores, target_class, results_path='./'):
    """
    Utilizes the package 'pytorch-grad-cam' which can be installed via "pip install grad-cam".
    Further encapsulates this package for use within our project, enhancing functionality and integration.
    """
    
    i=1
    for img_score in images_scores:
        score, img = img_score

        save_path = os.path.join(results_path, f'top{i}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        with open(os.path.join(save_path, 'probability.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)   
            writer.writerow([score])
        
        plt.imshow(img.squeeze().cpu(), cmap='gray')
        savepath = os.path.join(save_path, 'origin_image.png')
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        plt.close()

        img_tensor = img.unsqueeze(0)
        img_bw = img.squeeze().cpu().numpy()
        img_rgb = np.stack([img_bw, img_bw, img_bw], axis=-1) 
        img_normalized = np.float32(img_rgb) / 255

        for name, module in model.features.named_children():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d)):
                cam = GradCAM(model=model, target_layers=[module])
                targets = [ClassifierOutputTarget(target_class)]
                grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
                
                visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
                
                plt.imshow(visualization)
                plt.axis('off')
                savepath = os.path.join(save_path, f'gradcam_{name}_with_origin_image.png')
                plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
                plt.close()
                
                visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True, image_weight=0.0)
                
                plt.imshow(visualization)
                plt.axis('off')
                savepath = os.path.join(save_path, f'gradcam_{name}.png')
                plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
                plt.close()
        i+=1
