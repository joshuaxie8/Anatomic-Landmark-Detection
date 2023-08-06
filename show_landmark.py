import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PREDICTIONS_PATH='process_data/predictions.csv'
IMAGES_PATH='process_data/Test1Data'
LANDMARK_COL_START=1

image_idx=0

# Read predictions
predictions = pd.read_csv(PREDICTIONS_PATH)

def plot_image(idx, landmark=None):
    # image = plt.imread(IMAGES_PATH + '/' + predictions['image_file'][idx])
    image = Image.open(IMAGES_PATH + '/' + predictions['image_file'][idx])

    if landmark is not None:
        landmark = resize_landmark(landmark, image)
        plt.scatter(landmark[:,0], landmark[:,1], c = 'r', s = 5)

    plt.imshow(image)
    plt.show()


def reshape_landmark(landmark):
    landmark = landmark.reshape(-1, 2)
    return landmark

# Multiply x by image width and y by image height??
def resize_landmark(landmark, image):
    width, height = image.size

    landmark[:, 0] = landmark[:, 0] * width
    landmark[:, 1] = landmark[:, 1] * height
    print(f'resized: {landmark}')
    return landmark

image_landmark = predictions.iloc[image_idx, LANDMARK_COL_START:].values
image_landmark = reshape_landmark(image_landmark)
plot_image(image_idx, image_landmark)


