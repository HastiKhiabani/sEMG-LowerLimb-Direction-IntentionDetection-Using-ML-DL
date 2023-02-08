# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 08:50:00 2020

@author: Hasti                        
#Change the name of file you are reading from-corrospondinlgy change the name of file you are writing to
"""
import numpy as np
import pandas as pd
import feature_extraction as fe


number_of_cycle_train=4
number_of_cycle_test=1
num_subjects=1
num_windows=4000
#%%
def get_dataframe(examples_datasets, labels_datasets, number_of_cycle):
    participants_dataframes = []
    for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
        X = []
        Y = []

        for cycle in range(number_of_cycle):
            X.extend(participant_examples[cycle])
            Y.extend(participant_labels[cycle])
        data = {'examples': X,
                'labels': Y}
        df = pd.DataFrame(data)
        participants_dataframes.append(df)
    return participants_dataframes

#%%
def load_dataframe(number_of_cycle=4):  #########CHANGE#########
    
    participants_dataframes_train, participants_dataframes_test=[],[]
#    #Hasti
    datasets_train = np.load("../Dataset/Formatted_Dataset_ML/Windowed_Data_Hasti_train_235_9S.npy", encoding="bytes", allow_pickle=True)
    examples_datasets_train, labels_datasets_train = datasets_train
    participants_dataframes_train = get_dataframe(examples_datasets_train, labels_datasets_train, number_of_cycle=number_of_cycle_train)

    
    datasets_test = np.load("../Dataset/Formatted_Dataset_ML/Windowed_Data_Hasti_test_235_9S.npy",allow_pickle=True)
    examples_datasets_test, labels_datasets_test = datasets_test    
    participants_dataframes_test = get_dataframe(examples_datasets_test, labels_datasets_test, number_of_cycle=number_of_cycle_test)
    
    return participants_dataframes_train, participants_dataframes_test

#%%
def extract_fetures_from_dataset():
    train_dataset, test_dataset = load_dataframe()

    num_subj = num_subjects
    num_window = num_windows

    train_features = [[]]
    test_features = [[]]

    train_class = []
    test_class = []

    window = 0
    
    for s in range(len(train_dataset)):
        subj_data = train_dataset[s]
        print("Current subject trainset: ", s+1)
        for w in range(len(subj_data)):
            win_data = subj_data.values[w]
            # Collect the feature vector for window #
            train_features[window].append(fe.extract_features(win_data[0]))
            # Add list to list of list (make slot for next feature vector)
            train_features.append(list())
            train_class.append(win_data[1])
            window += 1
            

    train_features = np.array(train_features[:-1])
    train_class = np.array(train_class)
    
#    np.save("../Dataset/processed_dataset/FEATURES_train_10", train_features)
#    np.save("../Dataset/processed_dataset/CLASS_train_10", train_class)

    np.save("../Dataset/Formatted_Dataset_ML/FEATURES_train_10_235_9S", train_features)
    np.save("../Dataset/Formatted_Dataset_ML/CLASS_train_10_235_9S", train_class)
    
    window = 0 
    
    for s in range(len(test_dataset)):

        subj_data = test_dataset[s]
        print("Current subject testset: ", s+1)
        for w in range(len(subj_data)):
            win_data = subj_data.values[w]
            # Collect the feature vector for window #
            test_features[window].append(fe.extract_features(win_data[0]))
            # Add list to list of list (make slot for next feature vector)
            test_features.append(list())
            test_class.append(win_data[1])
            window += 1

    test_features = np.array(test_features[:-1])
    test_class = np.array(test_class)
    
    np.save("../Dataset/Formatted_Dataset_ML/FEATURES_test_10_235_9S", test_features)
    np.save("../Dataset/Formatted_Dataset_ML/CLASS_test_10_235_9S", test_class)
    
   
    
#%%
if __name__ == "__main__":
    extract_fetures_from_dataset()
