import torch
import torch.nn as nn
import os

from preprocessor import load_data
from cnn import ConvNet
from train import train_model
from test import test_model
from gradcam import runGradCam
from utils import plot_training_validation_loss

def run(layer_configs, drop_out=0.5, bool_bn=True, bool_xavier=True, activation=nn.LeakyReLU, loss=nn.CrossEntropyLoss, outdim=2, results_path='./'):
    """
    Encapsulates the CNN Model Initialization process, specifying the loss function and optimizer, conducting training and testing phases, and interpreting outputs using Grad-CAM.
    This design aids in performing sensitivity analysis, predicting return trends for subsequent periods (5, 20, 60 days), and calculating detailed return values.
    """
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    #Model Initialization
    model = ConvNet(layer_configs, dropout=drop_out, BN=bool_bn, xavier=bool_xavier, activation_fn=activation, out_dim=outdim)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Specify Loss Function and Optimizer
    criterion = loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # Training
    trained_model = train_model(model, train_loader, valid_loader, optimizer, criterion, device, n_epochs=10000, patience=2, results_path=os.path.join(results_path,'train_validation'))
    plot_training_validation_loss(working_dir=os.path.join(results_path,'train_validation'))

    # Test
    top5_down, top5_up = test_model(trained_model, test_loader, criterion, device, results_path=os.path.join(results_path,'test'))
    
    # Grad-CAM
    savepath = os.path.join(results_path, 'gradcam', 'down')
    runGradCam(trained_model, top5_down, 0, savepath)
    savepath = os.path.join(results_path, 'gradcam', 'up')
    if outdim==2:
        runGradCam(trained_model, top5_up, 1, savepath)
    elif outdim==1:
        runGradCam(trained_model, top5_up, 0, savepath)


if __name__ == '__main__':
   
    '''Baseline model I20/R20'''
    print("Baseline model I20/R20")
    transformations=None
    train_loader, valid_loader, test_loader = load_data(batch_size=128, num_workers=4, transform=transformations)
    layer_configs_baseline = [
    (64, (5, 3), (3, 1), (2, 1), (2, 1)),  # (out_channels, kernel_size, stride, dilation, max_pool)
    (128, (5, 3), (1, 1), (1, 1), (2, 1)),
    (256, (5, 3), (1, 1), (1, 1), (2, 1))
    ]
    run(layer_configs=layer_configs_baseline, results_path='./results/baseline')    
    
    '''Sensitivity to Model Structure, I20/R20'''
    print("Extension: Sensitivity to Model Structure, I20/R20")
    # Filters
    print("Extension: Filters(32)")
    layer_configs_sensitivity = [
        (32, (5, 3), (3, 1), (2, 1), (2, 1)),
        (64, (5, 3), (1, 1), (1, 1), (2, 1)),
        (128, (5, 3), (1, 1), (1, 1), (2, 1))
        ]  # As pointed out by Zeiler and Fergus (2014), learned features become more complex in deeper layers, so we follow the literature and increase the number of filters after each convolutional layer by a factor of two
    run(layer_configs=layer_configs_sensitivity, results_path='./results/Sensitivity_to_Model_Structure/Filters(32)')
    
    print("Extension: Filters(128)")
    layer_configs_sensitivity = [
        (128, (5, 3), (3, 1), (2, 1), (2, 1)), 
        (256, (5, 3), (1, 1), (1, 1), (2, 1)),
        (512, (5, 3), (1, 1), (1, 1), (2, 1))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/Sensitivity_to_Model_Structure/Filters(128)')
    
    # layers
    print("Extension: Layers(2)")
    layer_configs_sensitivity = [
        (64, (5, 3), (3, 1), (2, 1), (2, 1)),
        (128, (5, 3), (1, 1), (1, 1), (2, 1))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/Sensitivity_to_Model_Structure/layers(2)')
    
    print("Extension: Layers(4)")
    layer_configs_sensitivity = [
        (64, (5, 3), (3, 1), (2, 1), (2, 1)),
        (128, (5, 3), (1, 1), (1, 1), (2, 1)),
        (256, (5, 3), (1, 1), (1, 1), (2, 1)),
        (512, (5, 3), (1, 1), (1, 1), (2, 1))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/Sensitivity_to_Model_Structure/layers(4)')
    
    # dropout
    print("Extension: Dropout(0.0)")
    run(layer_configs=layer_configs_baseline, drop_out=0.0, results_path='./results/Sensitivity_to_Model_Structure/Dropout(0)')
    
    print("Extension: Dropout(0.25)")
    run(layer_configs=layer_configs_baseline, drop_out=0.25, results_path='./results/Sensitivity_to_Model_Structure/Dropout(0_25)')
    
    print("Extension: Dropout(0.75)")
    run(layer_configs=layer_configs_baseline, drop_out=0.75, results_path='./results/Sensitivity_to_Model_Structure/Dropout(0_75)')
    
    # BN
    print("Extension: BN(no)")
    run(layer_configs=layer_configs_baseline, bool_bn=False, results_path='./results/Sensitivity_to_Model_Structure/BN(no)')
    
    # xavier
    print("Extension: Xavier(no)")
    run(layer_configs=layer_configs_baseline, bool_xavier=False, results_path='./results/Sensitivity_to_Model_Structure/xavier(no)')
    
    # Activation
    print("Extension: Activation(ReLU)")
    run(layer_configs=layer_configs_baseline, activation=nn.ReLU, results_path='./results/Sensitivity_to_Model_Structure/Activation(ReLU)')
        
    # max-pool size
    print("Extension: Max-pool Size(2, 2)")
    layer_configs_sensitivity = [
        (64, (5, 3), (3, 1), (2, 1), (2, 2)),
        (128, (5, 3), (1, 1), (1, 1), (2, 2)),
        (256, (5, 3), (1, 1), (1, 1), (2, 2))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/Sensitivity_to_Model_Structure/maxpool(2_2)')
    
    # filersize
    print("Extension: FilterSize(3, 3)")
    layer_configs_sensitivity = [
        (64, (3, 3), (3, 1), (2, 1), (2, 1)),
        (128, (3, 3), (1, 1), (1, 1), (2, 1)),
        (256, (3, 3), (1, 1), (1, 1), (2, 1))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/Sensitivity_to_Model_Structure/FilterSize(3_3)')
    
    print("Extension: FilterSize(7, 3)")
    layer_configs_sensitivity = [
        (64, (7, 3), (3, 1), (2, 1), (2, 1)),
        (128, (7, 3), (1, 1), (1, 1), (2, 1)),
        (256, (7, 3), (1, 1), (1, 1), (2, 1))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/Sensitivity_to_Model_Structure/FilterSize(7_3)')
    
    # stride/dilation
    print("Extension: Stride/Dilation: (1,1)/(2,1)")
    layer_configs_sensitivity = [
        (64, (5, 3), (1, 1), (2, 1), (2, 1)),
        (128, (5, 3), (1, 1), (1, 1), (2, 1)),
        (256, (5, 3), (1, 1), (1, 1), (2, 1))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/Sensitivity_to_Model_Structure/Stride_Dilation(1_1_2_1)')
    
    print("Extension: Stride/Dilation: (3,1)/(1,1)")
    layer_configs_sensitivity = [
        (64, (5, 3), (3, 1), (1, 1), (2, 1)),
        (128, (5, 3), (1, 1), (1, 1), (2, 1)),
        (256, (5, 3), (1, 1), (1, 1), (2, 1))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/Sensitivity_to_Model_Structure/Stride_Dilation(3_1_1_1)')
    
    print("Extension: Stride/Dilation: (1,1)/(1,1)")
    layer_configs_sensitivity = [
        (64, (5, 3), (1, 1), (1, 1), (2, 1)),
        (128, (5, 3), (1, 1), (1, 1), (2, 1)),
        (256, (5, 3), (1, 1), (1, 1), (2, 1))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/Sensitivity_to_Model_Structure/Stride_Dilation(1_1_1_1)')

    # dilation
    print("Extension: Dilation: case1")
    layer_configs_sensitivity = [
        (64, (5, 3), (1, 1), (2, 1), (2, 1)),
        (128, (5, 3), (1, 1), (2, 1), (2, 1)),
        (256, (5, 3), (1, 1), (2, 1), (2, 1))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/dilation/case1')
    
    print("Extension: Dilation: case2")
    layer_configs_sensitivity = [
        (64, (5, 3), (1, 1), (1, 1), (2, 1)),
        (128, (5, 3), (1, 1), (2, 1), (2, 1)),
        (256, (5, 3), (1, 1), (3, 1), (2, 1))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/dilation/case2')

    print("Extension: Dilation: case3")
    layer_configs_sensitivity = [
        (64, (5, 3), (1, 1), (1, 1), (2, 1)),
        (128, (5, 3), (1, 1), (2, 2), (2, 1)),
        (256, (5, 3), (1, 1), (3, 3), (2, 1))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/dilation/case3')

    print("Extension: Dilation: case4")
    layer_configs_sensitivity = [
        (64, (5, 3), (1, 1), (2, 2), (2, 1)),
        (128, (5, 3), (1, 1), (2, 2), (2, 1)),
        (256, (5, 3), (1, 1), (2, 2), (2, 1))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/dilation/case4')
    
    print("Extension: Dilation: case5")
    layer_configs_sensitivity = [
        (64, (5, 3), (1, 1), (1, 1), (2, 1)),
        (128, (5, 3), (1, 1), (1, 1), (2, 1)),
        (256, (5, 3), (1, 1), (1, 1), (2, 1))
        ]
    run(layer_configs=layer_configs_sensitivity, results_path='./results/dilation/case5')
    
    '''Predict the return trend of different subsequent y-days'''
    print("Extension: Predict the return trend of different subsequent y-days")
    # I20/R5
    print("Extension: I20/R5")
    train_loader, valid_loader, test_loader = load_data(batch_size=128, num_workers=4, transform=transformations, labelName='Ret_5d')
    run(layer_configs=layer_configs_baseline, results_path='./results/y_days/I20_R5')
    
    print("Extension: I20/R60")
    train_loader, valid_loader, test_loader = load_data(batch_size=128, num_workers=4, transform=transformations, labelName='Ret_60d')
    run(layer_configs=layer_configs_baseline, results_path='./results/y_days/I20_R60')
    
    '''Predict the detailed return values, I20/R20'''
    print("Extension: Predict the detailed return values, I20/R20")
    train_loader, valid_loader, test_loader = load_data(batch_size=128, num_workers=4, transform=transformations, bool_regression=True)
    run(layer_configs=layer_configs_baseline, loss=nn.MSELoss, outdim=1, results_path='./results/detailed_return_values')
    
    