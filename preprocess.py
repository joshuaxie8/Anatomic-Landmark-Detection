import pandas as pd
import numpy as np
import os

NUM_IMAGES = 400
NUM_LANDMARKS = 19
ANNOTATIONS_PATH='./AnnotationsByMD/400_senior/'
SAVE_PATH='./process_data/'
SAVE_FILE_TRAIN='cepha_train.csv'
SAVE_FILE_TEST_1='cepha_val.csv'
TRAIN_IMAGES=150
TEST_1_IMAGES=150
TEST_2_IMAGES=100

# Image names are 001.bmp, 002.bmp, 003.bmp, ..., 400.bmp
ZFILL_NUM=3

FILES = os.listdir(ANNOTATIONS_PATH)


def read_image_annotation(filename):
    df = pd.read_csv(ANNOTATIONS_PATH + filename, header=None)
    df = df.iloc[:NUM_LANDMARKS]
    df.columns=['x', 'y']
    df = df.values
    return df

def read_all_annotations():
    annotations = []
    for filename in FILES:
        annotations.append(read_image_annotation(filename))
    
    annotations = np.array(annotations)
    return annotations

def reshape_annotations(annotations):
    return annotations.reshape(-1, NUM_LANDMARKS*2)

def split_df(df):
    df_train = df[:TRAIN_IMAGES]
    df_test_1 = df[TRAIN_IMAGES:TRAIN_IMAGES+TEST_1_IMAGES]
    df_test_2 = df[TRAIN_IMAGES+TEST_1_IMAGES:TRAIN_IMAGES+TEST_1_IMAGES+TEST_2_IMAGES]
    return df_train, df_test_1, df_test_2

def numpy_to_pandas(df):
    columns = []
    for i in range(NUM_LANDMARKS):
        columns.append('x'+str(i))
        columns.append('y'+str(i))

    df = pd.DataFrame(df, columns=columns)
    return df

def add_image_name(df, start, end):
    image_names = []
    for i in range(start, end):
        image_names.append(str(i).zfill(ZFILL_NUM)+'.bmp')

    df.insert(0, 'image_file', image_names)
    return df

df = read_all_annotations()
df = reshape_annotations(df)

print(df)
print(df.shape)

df_train, df_test_1, df_test_2 = split_df(df)

df_train = numpy_to_pandas(df_train)
df_test_1 = numpy_to_pandas(df_test_1)
df_test_2 = numpy_to_pandas(df_test_2)

df_train = add_image_name(df_train, 1, TRAIN_IMAGES+1)
df_test_1 = add_image_name(df_test_1, TRAIN_IMAGES+1, TRAIN_IMAGES+TEST_1_IMAGES+1)
df_test_2 = add_image_name(df_test_2, TRAIN_IMAGES+TEST_1_IMAGES+1, NUM_IMAGES+1)

df_train.to_csv(SAVE_PATH + SAVE_FILE_TRAIN, index=False)
df_test_1.to_csv(SAVE_PATH + SAVE_FILE_TEST_1, index=False)
# df_test_2.to_csv(SAVE_PATH + SAVE_FILE_TEST_2, index=False)

# Add image name per row in first column .bmp files
# 000-150
# 151-300
# 301-400

