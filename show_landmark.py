import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from PIL import Image

PREDICTIONS_PATH='process_data/predictions.csv'
IMAGES_PATH='process_data/Test1Data'
LABELS_PATH='process_data/cepha_val.csv'
LANDMARK_COL_START=1

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--imgNum", type=int, default=1)
config = parser.parse_args()

image_idx=config.imgNum

# Read predictions
predictions = pd.read_csv(PREDICTIONS_PATH)
labels = pd.read_csv(LABELS_PATH)

def plot_image(image, landmark=None, label=None):
    if landmark is not None:
        plt.scatter(landmark[:,0], landmark[:,1], c = 'r', s = 5, alpha=0.5)

    if label is not None:
        plt.scatter(label[:,0], label[:,1], c = 'g', s = 5, alpha=0.5)

    plt.imshow(image)
    plt.show()


def reshape_landmark(landmark):
    landmark = landmark.reshape(-1, 2)
    return landmark


# Multiply x by image width and y by image height??
def resize_landmark(landmark, image):
    width, height = image.size
    print(f'image size: {width}x{height}')
    landmark[:, 0] = landmark[:, 0] * (width-1)
    landmark[:, 1] = landmark[:, 1] * (height-1)
    return landmark

def get_labels(idx):
    image_landmark = labels.iloc[idx, LANDMARK_COL_START:].values
    return image_landmark

print(f'Open image {predictions["image_file"][image_idx]}')
image = Image.open(IMAGES_PATH + '/' + predictions['image_file'][image_idx])

image_landmark = predictions.iloc[image_idx, LANDMARK_COL_START:].values
image_landmark = reshape_landmark(image_landmark)
image_landmark = resize_landmark(image_landmark, image)

label = get_labels(image_idx)
label = reshape_landmark(label)

print(f'resized: {image_landmark}')
print(f'label: {label}')


plot_image(image, image_landmark, label)

