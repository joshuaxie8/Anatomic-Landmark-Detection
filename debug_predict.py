#!/usr/bin/env python3
from __future__ import print_function, division
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from dataLoader import Rescale, RandomCrop, ToTensor, LandmarksDataset
import models
import train
import lossFunction
import argparse
import pandas as pd
import torch
import numpy as np
import os

plt.ion()   # interactive mode

parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--landmarkNum", type=int, default=19)
parser.add_argument("--image_scale", default=(800, 640), type=tuple)
parser.add_argument("--use_gpu", type=int, default=0)
parser.add_argument("--spacing", type=float, default=0.1)
parser.add_argument("--R1", type=int, default=41)
parser.add_argument("--R2", type=int, default=41)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--data_enhanceNum", type=int, default=1)
parser.add_argument("--stage", type=str, default="train")
parser.add_argument("--saveName", type=str, default="test1")
parser.add_argument("--testName", type=str, default="30cepha100_fusion_unsuper.pkl")
parser.add_argument("--dataRoot", type=str, default="process_data/")
parser.add_argument("--supervised_dataset_train", type=str, default="cepha/")
parser.add_argument("--supervised_dataset_test", type=str, default="cepha/")
parser.add_argument("--unsupervised_dataset", type=str, default="cepha/")
parser.add_argument("--trainingSetCsv", type=str, default="cepha_train.csv")
parser.add_argument("--testSetCsv", type=str, default="cepha_val.csv")
parser.add_argument("--unsupervisedCsv", type=str, default="cepha_val.csv")
parser.add_argument("--numWorkers", type=int, default=12)
parser.add_argument("--modelPath", type=str, default="model/model.pth")
parser.add_argument("--predictionsPath", type=str, default="predictions.csv")

def main():
    print("=== Starting Debug Prediction ===")
    
    # Load model
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    config = parser.parse_args()
    
    print("Loading model...")
    model_ft = models.fusionVGG19(torchvision.models.vgg19_bn(pretrained=True), config).cuda(config.use_gpu)
    
    print("Loading model state...")
    model_ft.load_state_dict(torch.load(config.modelPath))
    print("Model loaded successfully!")
    
    print("Image scale:", config.image_scale)
    print("GPU:", config.use_gpu)

    # Load dataset
    print("Loading dataset...")
    transform_origin = torchvision.transforms.Compose([
                    Rescale(config.image_scale),
                    ToTensor()
                    ])

    val_dataset = LandmarksDataset(csv_file=config.dataRoot + config.testSetCsv,
                                                root_dir=config.dataRoot + config.supervised_dataset_test,
                                                transform=transform_origin,
                                                landmarksNum=config.landmarkNum
                                                )

    val_dataloader = []

    val_dataloader_t = DataLoader(val_dataset, batch_size=config.batchSize,
                            shuffle=False, num_workers=config.numWorkers)

    for data in val_dataloader_t:
        val_dataloader.append(data)

    print(f"Val length: {len(val_dataloader)}")

    dataloaders = {'val': val_dataloader}

    # Print model
    para_list = list(model_ft.children())

    print("model_ft.children() len", len(para_list))
    for idx in range(len(para_list)):
        print(idx, "-------------------->>>>", para_list[idx])

    model_ft = model_ft.cuda(config.use_gpu)

    # Predict on data
    print("Running predictions...")
    predictions = train.val(model_ft, dataloaders, config)
    print("Predictions shape:", predictions.shape)
    print("Predictions type:", type(predictions))
    print("Predictions dtype:", predictions.dtype)
    
    # Process predictions
    print("Processing predictions...")
    predictions = process_predictions(config.dataRoot + config.testSetCsv, predictions, config)
    print("Processed predictions shape:", predictions.shape)
    print("Processed predictions columns:", list(predictions.columns))
    
    # Save predictions
    output_path = config.dataRoot + config.predictionsPath
    print(f"Saving predictions to: {output_path}")
    
    try:
        predictions.to_csv(output_path, index=False, encoding='utf-8')
        print("Predictions saved successfully!")
        
        # Verify the file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"File created with size: {file_size} bytes")
        else:
            print("ERROR: File was not created!")
            
    except Exception as e:
        print(f"ERROR saving predictions: {e}")

# combine image name and resolution with predictions
def process_predictions(dataset_path, predictions, config):
    print(f"Reading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path, encoding='utf-8')
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")
    
    predictions = numpy_to_pandas(predictions, config)
    print(f"Numpy to pandas shape: {predictions.shape}")

    # insert image name in first column
    predictions.insert(0, 'image_file', df['image_file'])
    print(f"After inserting image_file: {predictions.shape}")

    return predictions

def numpy_to_pandas(predictions, config):
    print("Original predictions shape:", predictions.shape)
    
    # Reshape
    predictions = predictions.reshape(-1, config.landmarkNum*2)
    print("Reshaped predictions shape:", predictions.shape)

    # Convert pandas to numpy
    predictions = pd.DataFrame(predictions)
    
    columns = []
    for i in range(config.landmarkNum):
        columns.append('x'+str(i))
        columns.append('y'+str(i))

    predictions.columns = columns
    print("Final predictions shape:", predictions.shape)
    return predictions

if __name__ == "__main__":
    main() 