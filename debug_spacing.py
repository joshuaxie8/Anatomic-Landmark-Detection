import torch
import numpy as np
import pandas as pd
from dataLoader_preprocessed import LandmarksDataset, ToTensor
from torch.utils.data import DataLoader
import models
import torchvision
import utils

def debug_spacing():
    """
    Debug script to understand the spacing issue and determine correct spacing parameter.
    """
    
    # Configuration
    config = type('Config', (), {
        'batchSize': 1,
        'landmarkNum': 19,
        'image_scale': (800, 640),
        'use_gpu': 0,
        'spacing': 0.1,  # Current spacing
        'R1': 41,
        'R2': 41,
        'epochs': 400,
        'dataRoot': 'process_data/',
        'supervised_dataset_test': 'Test1Data_preprocessed/',
        'testSetCsv': 'cepha_val.csv',
        'numWorkers': 4,
        'modelPath': 'model/model_best.pth'
    })()
    
    print("=== SPACING DEBUG ANALYSIS ===")
    print(f"Current spacing: {config.spacing} mm/pixel")
    print(f"Original image dimensions: 1934 x 2399 pixels")
    print(f"Model input dimensions: {config.image_scale}")
    
    # Load a small dataset for analysis
    transform = ToTensor()
    val_dataset = LandmarksDataset(
        csv_file=config.dataRoot + config.testSetCsv,
        root_dir=config.dataRoot + config.supervised_dataset_test,
        transform=transform,
        landmarksNum=config.landmarkNum
    )
    
    val_dataloader = DataLoader(val_dataset, batch_size=config.batchSize,
                               shuffle=False, num_workers=config.numWorkers)
    
    print(f"\nDataset size: {len(val_dataset)} images")
    
    # Analyze a few samples
    sample_count = min(5, len(val_dataset))
    pixel_errors = []
    
    for i, data in enumerate(val_dataloader):
        if i >= sample_count:
            break
            
        labels = data['landmarks']  # Ground truth (normalized 0-1)
        
        # Convert to original pixel coordinates
        labels_pixel = labels.clone()
        labels_pixel[:, :, 0] = labels_pixel[:, :, 0] * 1934
        labels_pixel[:, :, 1] = labels_pixel[:, :, 1] * 2399
        
        print(f"\nSample {i+1}:")
        print(f"  Normalized coordinates (first 3 landmarks):")
        print(f"    {labels[0, :3, :].numpy()}")
        print(f"  Pixel coordinates (first 3 landmarks):")
        print(f"    {labels_pixel[0, :3, :].numpy()}")
        
        # Calculate some example distances between landmarks
        for j in range(min(3, config.landmarkNum-1)):
            dist_pixels = np.sqrt(
                (labels_pixel[0, j+1, 0] - labels_pixel[0, j, 0])**2 + 
                (labels_pixel[0, j+1, 1] - labels_pixel[0, j, 1])**2
            )
            dist_mm_current = dist_pixels * config.spacing
            print(f"  Distance between landmarks {j} and {j+1}: {dist_pixels:.1f} pixels = {dist_mm_current:.2f} mm")
    
    # Test different spacing values
    print(f"\n=== TESTING DIFFERENT SPACING VALUES ===")
    
    # Load model if available
    try:
        model_ft = models.fusionVGG19(torchvision.models.vgg19_bn(pretrained=True), config).cuda(config.use_gpu)
        model_ft.load_state_dict(torch.load(config.modelPath))
        model_ft.eval()
        print("Model loaded successfully")
        
        # Get predictions for first sample
        data = next(iter(val_dataloader))
        inputs = data['image'].cuda(config.use_gpu)
        labels = data['landmarks'].cuda(config.use_gpu)
        
        with torch.no_grad():
            heatmaps = model_ft(inputs)
            predicted_landmarks = utils.regression_voting(heatmaps, config.R2).cuda(config.use_gpu)
            dev = utils.calculate_deviation(predicted_landmarks.detach(), labels.detach())
        
        print(f"\nPrediction analysis:")
        print(f"  Raw pixel errors (first 5 landmarks): {dev[0, :5].cpu().numpy()}")
        
        # Test different spacing values
        spacing_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        for spacing in spacing_values:
            dev_mm = dev * spacing
            sdr, sd, mre = utils.get_statistical_results(dev_mm, config)
            
            print(f"\nSpacing {spacing} mm/pixel:")
            print(f"  MRE: {torch.mean(mre).item():.3f} mm")
            print(f"  SD: {torch.mean(sd).item():.3f} mm")
            print(f"  SDR [1mm, 2mm, 2.5mm, 3mm, 4mm]: {torch.mean(sdr, 0).cpu().numpy()}")
            
            # Check if SDR is reasonable (not all 1.0)
            if torch.all(torch.mean(sdr, 0) == 1.0):
                print(f"       WARNING: All SDR values are 1.0 - spacing might be too small!")
            elif torch.all(torch.mean(sdr, 0) == 0.0):
                print(f"       WARNING: All SDR values are 0.0 - spacing might be too large!")
            else:
                print(f"       SDR values look reasonable")
        
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Running spacing analysis without model predictions")
    
    # Provide recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    print("1. For cephalometric X-rays, typical pixel spacing is 0.1-0.2 mm/pixel")
    print("2. If SDR shows all 1.0, the spacing is too small (errors appear smaller than they are)")
    print("3. If SDR shows all 0.0, the spacing is too large (errors appear larger than they are)")
    print("4. Look for spacing values that give reasonable SDR distributions")
    print("5. You may need to consult the original imaging setup documentation for exact spacing")

if __name__ == "__main__":
    debug_spacing() 