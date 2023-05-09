# Write Data loaders, training procedure and validation procedure in this file.
import os
import numpy as np
import pandas as pd
import csv

# Load the dataset
data_path = os.getcwd() + "/dataset/input"
label_path = os.getcwd() + "/dataset/label"

train_x = []
val_x = []
test_x = []
train_labels = []
val_labels = []
test_labels = []

for file in os.listdir(data_path):
    print(file)
    data_file_path = os.path.join(data_path, file)
    label_file_path = os.path.join(label_path, file)
    with open(data_file_path, 'r') as f1, open(label_file_path, 'r') as f2:
        data = f1.readlines()
        labels = f2.readlines()

        max_len = 5*15
        for i in range(len(data)):
            data[i] = data[i].split()
            data[i] = [float(x) for x in data[i]]
            data[i] += [0]*(max_len - len(data[i])) 
            labels[i] = labels[i].split()
            labels[i] = [float(x) for x in labels[i]]
            labels[i] += [0]*(max_len - len(labels[i])) 

        indices = np.arange(len(data))
        np.random.shuffle(indices)
        
        #Split into train, val, test
        data = np.array(data)
        data = data[indices].tolist()
        data_len = len(data)
        train_x += data[:int(0.7*data_len)]
        val_x += data[int(0.7*data_len):int(0.9*data_len)]
        test_x += data[int(0.9*data_len):]

        labels = np.array(labels)
        labels = labels[indices].tolist()
        labels_len = len(labels)
        train_labels += labels[:int(0.7*labels_len)]
        val_labels += labels[int(0.7*labels_len):int(0.9*labels_len)]
        test_labels += labels[int(0.9*labels_len):]

train_x = pd.DataFrame(train_x)
val_x = pd.DataFrame(val_x)
test_x = pd.DataFrame(test_x)
train_labels = pd.DataFrame(train_labels)
val_labels = pd.DataFrame(val_labels)
test_labels = pd.DataFrame(test_labels)

def make_new_dataframe(org_df):
    def combine_cols(row):
        return [row[col] for col in cols_to_combine]
    new_df = pd.DataFrame()
    for i in range(15):
        cols_to_combine = list(org_df.columns[i:i+5])
        new_df["col"+str(i)] = org_df.apply(combine_cols, axis=1)
    return new_df

train_x = make_new_dataframe(train_x)
val_x = make_new_dataframe(val_x)
test_x = make_new_dataframe(test_x)
train_labels = make_new_dataframe(train_labels)
val_labels = make_new_dataframe(val_labels)
test_labels = make_new_dataframe(test_labels)

train_x.to_csv("train_x.csv",index=False, header=False)
val_x.to_csv("val_x.csv",index=False, header=False)
test_x.to_csv("test_x.csv",index=False, header=False)
train_labels.to_csv("train_y.csv",index=False, header=False)
val_labels.to_csv("val_y.csv",index=False, header=False)
test_labels.to_csv("test_y.csv",index=False, header=False)


