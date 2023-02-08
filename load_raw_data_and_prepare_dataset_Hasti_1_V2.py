# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:33:24 2020

@author: Hasti                  #CHANGE ROW 69 TO CHANGE NUMBER OF SUBJECTS TO ADD!
"""

import os
import numpy as np
#%%

window_size=260
size_non_overlap=25
tests={"AnkleDorsiflexion":0,"AnklePlantarflexion":1,"AnkleEversion":2,"AnkleInversion":3,"AnkleLateralRotation":4, "AnkleMedialRotation":5, "KneeAbduction":6, "KneeAdduction":7, "KneeExtension":8, "KneeFlexion":9, "RestPosition":10}
number_of_gestures=len(tests) 
DAQ_filename = "DAQ_outputs.txt"
number_of_canals=8
number_of_trials_train=4*len(tests)
number_of_trials_test=1*len(tests)
num_cycles=4

#ti change number of subjects go to if statement and change the subject number that you want to break before!

cutoff_Low=20
cutoff_High=450
sampling_freq=1000
filter_order=4

#%%
#from PrepareAndLoadData.prepare_dataset_utils import butter_bandpass_filter
from prepare_dataset_utils import butter_bandpass_filter


def format_examples(emg_examples, window_size=window_size, size_non_overlap=size_non_overlap):
    examples_formatted = []
    example = []
    for emg_vector in emg_examples:
        if len(example) == 0:
            example = emg_vector
        else:
            example = np.row_stack((example, emg_vector))

        if len(example) >= window_size:
            # The example is of the shape TIME x CHANNEL. Make it of the shape CHANNEL x TIME
            example = example.transpose()
            # Go over each channel and bandpass filter it between 20 and 495 Hz.
            example_filtered = []
            for channel in example:
                channel_filtered = butter_bandpass_filter(channel, lowcut=cutoff_Low, highcut=cutoff_High, fs=sampling_freq, order=filter_order)
                example_filtered.append(channel_filtered)
            # Add the filtered example to the list of examples to return and transpose the example array again to go
            # back to TIME x CHANNEL
            examples_formatted.append(example_filtered)
            example = example.transpose()
            # Remove part of the data of the example according to the size_non_overlap variable
            example = example[size_non_overlap:]

    return examples_formatted

#%%
def get_data_and_process_it_from_file(get_train_data, path, number_of_gestures=number_of_gestures, number_of_cycles=num_cycles, window_size=window_size,
                                      size_non_overlap=size_non_overlap):
#    examples_datasets, labels_datasets = [], []
    examples_participants, labels_participant = [], []
    train_or_test_str = "train" if get_train_data else "test"

    participants_directories = os.listdir(path)
    for participant_directory in participants_directories:
        if participant_directory == "S90":  #change to number of subjects you  want+1
            break
        else:
            examples_trials=[] 
            labels_trials=[]
            
            trials_directories = os.listdir(path + '/' + participant_directory + '/' + train_or_test_str)
            for trial_directory in trials_directories:
                
                labels = []
                examples = []
                
                for dirs in os.listdir(path + '/' + participant_directory + '/' + train_or_test_str + '/' + trial_directory):
                    print("Preparing data of: " + participant_directory + " Set:" + train_or_test_str + " from " + trial_directory + " " + dirs)                
                    DAQ_data = np.stack(np.loadtxt(open(path + '/' + participant_directory + '/' + train_or_test_str + '/' + trial_directory + "/" + dirs + "/" + DAQ_filename).readlines()[:-1], skiprows=2, delimiter=',',usecols=(0,1,2,3,4,5,6,7,8)))
    #                EMG_Data = DAQ_data[:,0:number_of_canals]
                    if dirs.find("_")==-1:
                        name=dirs
                        EMG_Data = DAQ_data[1500:6500,0:number_of_canals]
                    else:
                        name=dirs[:dirs.find("_")]
                        EMG_Data = DAQ_data[1500:6600,0:number_of_canals]                                    
                    data_read_from_file = np.array(EMG_Data, dtype=np.float32)
                    examples_formatted = format_examples(data_read_from_file, window_size=window_size, size_non_overlap=size_non_overlap)
                    examples.extend(examples_formatted)
                    labels.extend((tests[name]) + np.zeros(len(examples_formatted))) 
                                       
                examples_trials.append(examples)
                labels_trials.append(labels)
    
            examples_participants.append(examples_trials)
            labels_participant.append(labels_trials)

    return examples_participants, labels_participant

def read_data(path, number_of_gestures= number_of_gestures, number_of_cycles=num_cycles, window_size=window_size+1, size_non_overlap=size_non_overlap):
    print("Loading and preparing datasets...")
    'Get and process the train data'
    print("Taking care of the training data...")
    list_dataset_train_emg, list_labels_train_emg = get_data_and_process_it_from_file(get_train_data=True, path=path,
                                                                                      number_of_gestures=
                                                                                      number_of_gestures,
                                                                                      number_of_cycles=num_cycles,
                                                                                      window_size=window_size,
                                                                                      size_non_overlap=size_non_overlap)
    np.save("../Dataset/Formatted_Dataset_ML/Windowed_Data_Hasti_train_235_9S", (list_dataset_train_emg, list_labels_train_emg))
    #change #subjects above aswell!
    print("Finished with the training data...")


#    'Get and process the test data'
    print("Starting with the test data...")
    list_dataset_train_emg, list_labels_train_emg = get_data_and_process_it_from_file(get_train_data=False, path=path,
                                                                                      number_of_gestures=
                                                                                      number_of_gestures,
                                                                                      number_of_cycles=num_cycles,
                                                                                      window_size=window_size,
                                                                                      size_non_overlap=size_non_overlap)
    np.save("../Dataset/Formatted_Dataset_ML/Windowed_Data_Hasti_test_235_9S", (list_dataset_train_emg, list_labels_train_emg))
    #change #subjects above aswell!
    print("Finished with the test data")
    

if __name__ == '__main__':
#    read_data(path="../Dataset/Hasti/processed_dataset_Hasti/Windowed_Test_Hasti, window_size=151)
    read_data(path="../Dataset/Main_Dataset", window_size=window_size) #changed from (window_size=size_non_overlap+1)