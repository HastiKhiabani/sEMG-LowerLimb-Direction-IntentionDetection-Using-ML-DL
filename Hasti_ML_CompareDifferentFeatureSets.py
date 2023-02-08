# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 23:19:41 2020

@author: Hasti
"""
#import HastiGetDatafromPythonScript
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import Models_Modified as Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


from load_raw_data_and_prepare_dataset_Hasti_1_V2 import read_data
from GetData_GetFuture_Hasti_2 import extract_fetures_from_dataset
#%%
gestures = {"AnkleDorsiflexion","AnklePlantarflexion", "AnkleEversion", "AnkleInversion", "AnkleLateralRotation","AnkleMedialRotation","KneeAbduction","KneeAdduction", "KneeExtension","KneeFlexion", "RestPosition"}    #["Clench","Left","Right","Up","Rest"]
num_of_classes=len(gestures)
# gestures_brief = {"DF","PF", "Ever", "Inv", "AnLRot","AnMRot","Ab","Ad", "KnExt","KnFlex", "Rest"}    #["Clench","Left","Right","Up","Rest"]
Clf_types=['KNN', 'Lda', 'Qda', 'SVM', 'DecTree', 'GNB' ,'RandForest', 'Ens_Bag', 'Ens_Ada', 'Ens_GBDT']


#%% Deminsionality Reduction
    
def get_dimensionality_reduction(componentstokeep, dataset):
    pca = PCA(n_components=componentstokeep, svd_solver='full')  #(n_components=n_components, svd_solver='full'or'randomized',whiten=True),
    pca.fit(dataset)
    dataset_aftertraining = pca.transform(dataset)
    pca_variance = pca.explained_variance_
    plt.figure(figsize=(8, 6))
    plt.bar(range(componentstokeep), pca_variance, alpha=0.5, align='center', label='individual variance')
    plt.legend()
    plt.ylabel('Variance ratio')
    plt.xlabel('Principal components')
    plt.show()
    return (dataset_aftertraining)

#%%
def confusion_matrix(pred, Y, number_class=num_of_classes):                #change!
    confusion_matrice = []
    for x in range(0, number_class):
        vector = []
        for y in range(0, number_class):
            vector.append(0) 
        confusion_matrice.append(vector)
    for prediction, real_value in zip(pred, Y):
        prediction = int(prediction)
        real_value = int(real_value)
        confusion_matrice[prediction][real_value] += 1
    return np.array(confusion_matrice)

def plot_confusion(classifier, X_test, y_test, name, TrainedFolderName):
    # Plot non-normalized confusion matrix
    titles_options = [("Normalized confusion matrix", 'true', 'Normalized'),
                      ("Confusion matrix, without normalization", None, 'Non_Normalized')]
    for title, normalize, header in titles_options:
        fig, ax = plt.subplots(figsize=(15, 15))
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=gestures,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize,
                                     ax=ax)
        disp.ax_.set_title(title)
        plt.xticks(rotation=45)
        print(title)
        print(disp.confusion_matrix)
        plt.savefig(TrainedFolderName + '/plots/%s' %name + '_%s.png' %header)
        plt.show()
        plt.close()
    return 

#%%
def calculate_fitness( phase, PCA, PCA_componentstokeep, Features_train, labels_train, Features_test, labels_test, TrainedFolderName):      
    
    Features_train = np.reshape(Features_train,(len(Features_train),len(Features_train[0][0])))
    Features_test = np.reshape(Features_test,(len(Features_test),len(Features_test[0][0]))) 
    
    # X_train, X_validation, y_train, y_validation = train_test_split(Features_train, labels_train, test_size=0.2, random_state=4)
    # print ('Train set:', X_train.shape,  y_train.shape)
    # print ('Test set:', X_validation.shape,  y_validation.shape)


#%% Add Dimensionality reduction
    if PCA == "ON":
        Features_train_afterPCA = get_dimensionality_reduction(PCA_componentstokeep, Features_train)
        print(Features_train_afterPCA.shape)
        
        Features_test_afterPCA = get_dimensionality_reduction(PCA_componentstokeep, Features_test)
        print(Features_test_afterPCA.shape)
    
        Features_train = Features_train_afterPCA
        Features_test = Features_test_afterPCA
    else:
        pass
        
#%% Apply Different Classifiers (if you have already trained it you can chose test)   
    Clf_types=['KNN', 'Lda', 'Qda', 'SVM', 'DecTree', 'GNB' ,'RandForest', 'Ens_Bag', 'Ens_Ada', 'Ens_GBDT']
    Results_Train=[]
    Results_Validation=[]
    Clf_results_Validation=[[]] * 10
    Clf_results_Validation[0].append(0)
    Clf_score_test=[]
    
    if "training" in phase:    
        KNN=[0]
        Lda=[0]
        Qda=[0]
        SVM=[0]
        DT=[0]
        GNB=[0]
        RF=[0]
        Bag=[0]
        Ada=[0]
        GBDT=[0]
        
        KNN_Classifier, CV_AvgAccuracy_KNN, GridSearch_results_Knn = Model.train_model_KNN(Features_train, labels_train, TrainedFolderName)
        KNN = [CV_AvgAccuracy_KNN]
        
        LDA_Classifier, CV_AvgAccuracy_LDA, GridSearch_results_LDA = Model.train_model_LDA(Features_train, labels_train, TrainedFolderName)
        Lda = [CV_AvgAccuracy_LDA]
    
        QDA_Classifier, CV_AvgAccuracy_QDA, GridSearch_results_QDA = Model.train_model_QDA(Features_train, labels_train, TrainedFolderName)
        Qda = [CV_AvgAccuracy_QDA]
        
        SVM_Classifier, CV_AvgAccuracy_SVM, GridSearch_results_SVM = Model.train_model_SVM(Features_train, labels_train, TrainedFolderName)
        SVM = [CV_AvgAccuracy_SVM]
        
        DT_Classifier, CV_AvgAccuracy_DT, GridSearch_results_DT = Model.train_model_DT(Features_train, labels_train, TrainedFolderName)
        DT = [CV_AvgAccuracy_DT]
        
        GNB_Classifier, CV_AvgAccuracy_GNB, GridSearch_results_GNB = Model.train_model_GNB(Features_train, labels_train, TrainedFolderName)
        GNB = [CV_AvgAccuracy_GNB]   
        
        RF_Classifier, CV_AvgAccuracy_RF, GridSearch_results_RF = Model.train_model_RF(Features_train, labels_train, TrainedFolderName)
        RF = [CV_AvgAccuracy_RF]   
        
        Bag_Classifier, CV_AvgAccuracy_Bag, GridSearch_results_Bag = Model.train_model_Bag(Features_train, labels_train, TrainedFolderName)
        Bag = [CV_AvgAccuracy_Bag]      
    
        Ada_Classifier, CV_AvgAccuracy_Ada, GridSearch_results_Ada = Model.train_model_Ada(Features_train, labels_train, TrainedFolderName)
        Ada = [CV_AvgAccuracy_Ada]  
    
        GBDT_Classifier, CV_AvgAccuracy_GBDT, GridSearch_results_GBDT = Model.train_model_GBDT(Features_train, labels_train, TrainedFolderName)
        # GBDT = [CV_AvgAccuracy_GBDT]  
        
        Clf_results_Validation = [KNN[0], Lda[0] , Qda[0], SVM[0], DT[0], GNB[0], RF[0], Bag[0], Ada[0], GBDT[0]]
        
        Results_Validation=pd.DataFrame(
            {'Classifier': Clf_types,
              'model_validation_score': Clf_results_Validation})
        Results_Validation['model_validation_score'] = Results_Validation['model_validation_score'].apply(lambda x: x*100)
        
        # Results_Validation_F1score=pd.DataFrame(
        #     {'Classifier': Clf_types,
        #       'model_validation_F1score': Clf_results_Validation_F1score})
        # Results_Validation_F1score['model_validation_F1score'] = Results_Validation_F1score['model_validation_F1score'].apply(lambda x: x*100)
        
        # Results_Validation_Jaccord=pd.DataFrame(
        #     {'Classifier': Clf_types,
        #       'model_validation_Jaccordscore': Clf_results_Validation_Jaccard})
        # Results_Validation_Jaccord['model_validation_Jaccordscore'] = Results_Validation_Jaccord['model_validation_Jaccordscore'].apply(lambda x: x*100)
        
    else: 
        pass

#%% testset examination and Confusion Matrix (you have this one compacted in the next section)         
    if "testing" in phase: 

        KNN=[0.7290233202059829]
        Lda=[0.6215158062149729]
        Qda=[0.45321291054341845]
        SVM=[0.7382503973152347]
        DT=[0.652190474163561]
        GNB=[0.39966829703526763]
        RF=[0.7920434929770462]
        Bag=[0.7979768744461995]
        Ada=[0.3554038168498554]
        GBDT=[0.318209139456298]
        
        #load the model from disk
        name=['Knnpickle_file','Ldapickle_file','Qdapickle_file','svmpickle_file','DTpickle_file','GNBpickle_file','RFpickle_file','Bagpickle_file','Adapickle_file','GBDTpickle_file'] #
        # name=['Ldapickle_file'] #
          
        for clf in name:
            loaded_model = pickle.load(open(TrainedFolderName+ '/' +clf, 'rb')) 
            model=loaded_model.best_estimator_
            result_score = model.score(Features_test, labels_test)
            # result_score_F1 = loaded_model.f1_score(Features_test, labels_test)
            # result_score_Jaccord = loaded_model.jaccard_score(Features_test, labels_test)
            print("Accuracy on TestSet from %s = %f" %(clf, 100*result_score))
            Clf_score_test.append(result_score*100)  #result_score_tot is removed
            #Save Confusion-Matrix
            # result = loaded_model.predict(Features_test)
            # confusion_mat= confusion_matrix(result, labels_test, number_class=num_of_classes)
            plot_confusion(loaded_model, Features_test, labels_test, clf, TrainedFolderName)
            
        Results_test=pd.DataFrame(
            {'Classifier': Clf_types,
              'model_score': Clf_score_test})
        Results_test.sort_values(by='model_score', ascending=True) #not working!
    
    
    return Clf_results_Validation, Clf_score_test       
#%%
if __name__ == '__main__':    
    # Comment between here
    '''
    list_dataset_train_emg, list_labels_train_emg, list_dataset_test_emg, list_labels_test_emg = read_data(path="../Dataset/Main_Dataset", window_size=260, number_of_subjects_to_add='S3')
    extract_fetures_from_dataset(path_windowed_train="../Dataset/Formatted_Dataset_ML/Windowed_Data_Hasti_train_235_9S.npy", 
                                  path_windowed_test="../Dataset/Formatted_Dataset_ML/Windowed_Data_Hasti_test_235_9S.npy")    
    # And here if the pre-training dataset was already processed and saved and features are ectracted!
    # Comment between here
    '''
    
#%% 10 Features
    Features_train_9S_10 = np.load("../Dataset/Formatted_Dataset_ML/FEATURES_train_10_235_9S.npy", encoding="bytes", allow_pickle=True)
    labels_train_9S_10 = np.load("../Dataset/Formatted_Dataset_ML/CLASS_train_10_235_9S.npy", encoding="bytes", allow_pickle=True)
    Features_test_9S_10 = np.load("../Dataset/Formatted_Dataset_ML/FEATURES_test_10_235_9S.npy", encoding="bytes", allow_pickle=True)
    labels_test_9S_10 = np.load("../Dataset/Formatted_Dataset_ML/CLASS_test_10_235_9S.npy", encoding="bytes", allow_pickle=True)

    TrainedFolderName='results_10_235window_Stotal_CompareFeatures_9S'
    Results_Validation_accuracyscore_9S_10, Results_test_accuracyscore_9S_10 = calculate_fitness("testing", "OFF", 0,
                                                                                                  Features_train_9S_10, labels_train_9S_10, Features_test_9S_10, labels_test_9S_10, TrainedFolderName)
    
    # componentstokeep_10 = int(1* Features_train_9S_10.shape[2])    
    # TrainedFolderName='results_10PCA_235window_Stotal_CompareFeatures_9S'
    # Results_Validation_accuracyscore_9S_10_PCA, Results_test_accuracyscore_9S_10_PCA = calculate_fitness("testing", "ON", componentstokeep_10,
    #                                                                                                       Features_train_9S_10, labels_train_9S_10, Features_test_9S_10, labels_test_9S_10, TrainedFolderName)

#%% 15 Features
    Features_train_9S_15 = np.load("../Dataset/Formatted_Dataset_ML/FEATURES_train_15_235_9S.npy", encoding="bytes", allow_pickle=True)
    labels_train_9S_15 = np.load("../Dataset/Formatted_Dataset_ML/CLASS_train_15_235_9S.npy", encoding="bytes", allow_pickle=True)
    Features_test_9S_15 = np.load("../Dataset/Formatted_Dataset_ML/FEATURES_test_15_235_9S.npy", encoding="bytes", allow_pickle=True)
    labels_test_9S_15 = np.load("../Dataset/Formatted_Dataset_ML/CLASS_test_15_235_9S.npy", encoding="bytes", allow_pickle=True)

    TrainedFolderName='results_15_235window_Stotal_CompareFeatures_9S'
    Results_Validation_accuracyscore_9S_15, Results_test_accuracyscore_9S_15 = calculate_fitness("testing", "OFF", 0,
                                                                                                  Features_train_9S_15, labels_train_9S_15, Features_test_9S_15, labels_test_9S_15, TrainedFolderName)
    
    # componentstokeep_15 = int(0.8* Features_train_9S_15.shape[2])
    # TrainedFolderName='results_15PCA_235window_Stotal_CompareFeatures_9S'
    # Results_Validation_accuracyscore_9S_15_PCA, Results_test_accuracyscore_9S_15_PCA = calculate_fitness("testing", "ON", componentstokeep_15,
    #                                                                                                       Features_train_9S_15, labels_train_9S_15, Features_test_9S_15, labels_test_9S_15, componentstokeep_15, TrainedFolderName)


#%%
       
    Results=pd.DataFrame(
        {'Classifier': Clf_types,
          'model_validation_accscore_9S_10Features': Results_Validation_accuracyscore_9S_10,
          'model_test_accscore_9S_10Features': Results_test_accuracyscore_9S_10,
        
          # 'model_validation_accscore_9S_10Features_PCA': Results_Validation_accuracyscore_9S_10_PCA,
          # 'model_test_accscore_9S_10Features_PCA': Results_test_accuracyscore_9S_10_PCA,
         
          'model_validation_accscore_9S_15Features': Results_Validation_accuracyscore_9S_15,
          'model_test_accscore_9S_15Features': Results_test_accuracyscore_9S_15, 
        
          # 'model_validation_accscore_9S_15Features_PCA': Results_Validation_accuracyscore_9S_15_PCA,
          # 'model_test_accscore_9S_15Features_PCA': Results_test_accuracyscore_9S_15_PCA
          }
        )

#%%
    CompareFeaturesFolderName= 'results_235window_Stotal_CompareFeatures_9S'
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(CompareFeaturesFolderName + '/output.xlsx', engine='xlsxwriter')
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    Results.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    # Results = pd.read_pickle('HastiTry') 

#%%    
    x = np.arange(len(Clf_types))  # the label locations
    width = 0.15  # the width of the bars
    
    # fig, ax = plt.subplots(figsize=(10,5))
    # rects1 = ax.bar(x - (3*width)/2, Results['model_train_accscore_9S_10Features'], width, color='#FFA07A', label='TD')
    # rects2 = ax.bar(x - (width)/2, Results['model_train_accscore_9S_10Features_PCA'], width, color='#FF6347',  label='TD_PCA')
    # rects3 = ax.bar(x + (width)/2, Results['model_train_accscore_9S_15Features'], width, color='#FF0000', label='Enhanced-TD')
    # rects4 = ax.bar(x + (3*width)/2, Results['model_train_accscore_9S_15Features_PCA'], width, color='#8B0000', label='Enhanced-TD_PCA')
    # ax.set_title('Model accuracy Score for different Classifiers on TrainSet')
    # ax.set_ylabel('Scores')
    # ax.set_xticks(x)#(rotation=90)
    # ax.set_xticklabels(Results['Classifier'])
    # fig.tight_layout()
    # ax.legend()
    # plt.grid(axis='y')
    # plt.show()
    # fig.savefig(CompareFeaturesFolderName + '/plots/Results_TrainSet_Compare_accuracyscore.png')
    
    
    fig, ax = plt.subplots(figsize=(10,5))
    rects1 = ax.bar(x - (3*width)/2, Results['model_validation_accscore_9S_10Features'], width, color='#FFA07A', label='TD')
    # rects2 = ax.bar(x - width/2, Results['model_validation_accscore_9S_10Features_PCA'], width, color='#FF6347', label='TD_PCA')
    rects3 = ax.bar(x + (width)/2, Results['model_validation_accscore_9S_15Features'], width, color='#FF0000', label='Enhanced-TD')
    # rects4 = ax.bar(x + (3*width)/2, Results['model_validation_accscore_9S_15Features_PCA'], width, color='#8B0000', label='Enhanced-TD_PCA')
    ax.set_title('Model accuracy Score for different Classifiers on ValidationSet')
    ax.set_ylabel('Scores')
    ax.set_xticks(x)#(rotation=90)
    ax.set_xticklabels(Results['Classifier'])
    fig.tight_layout()
    ax.legend()
    plt.grid(axis='y')
    plt.show()
    fig.savefig(CompareFeaturesFolderName + '/plots/Results_ValidationSet_Compare_accuracyscore.png')
    
    
    fig, ax = plt.subplots(figsize=(10,5))
    rects1 = ax.bar(x - (3*width)/2, Results['model_test_accscore_9S_10Features'], width, color='#FFA07A', label='TD')
    # rects2 = ax.bar(x - width/2, Results['model_test_accscore_9S_10Features_PCA'], width, color='#FF6347', label='TD_PCA')
    rects3 = ax.bar(x + (width)/2, Results['model_test_accscore_9S_15Features'], width, color='#FF0000', label='Enhanced-TD')
    # rects4 = ax.bar(x + (3*width)/2, Results['model_test_accscore_9S_15Features_PCA'], width, color='#8B0000', label='Enhanced-TD_PCA')
    ax.set_title('Model accuracy Score for different Classifiers on TestSet')
    ax.set_ylabel('Scores')
    ax.set_xticks(x)#(rotation=90)
    ax.set_xticklabels(Results['Classifier'])
    fig.tight_layout()
    ax.legend()
    plt.grid(axis='y')
    plt.show()
    fig.savefig(CompareFeaturesFolderName + '/plots/Results_TestSet_Compare_accuracyscore.png')
    
    # Results.to_excel(CompareFeaturesFolderName + '/output.xlsx', sheet_name='Sheet_name_1')    
    # Results.to_excel("output_CompareAll_Features.xlsm")
    
    numpy_array = Results.to_numpy()
    np.savetxt(CompareFeaturesFolderName + "/test_file.txt", numpy_array, fmt = "%d")
# #     Results.to_excel("output_CompareAll_Features.xlsm")
# #     # with open("results/Final_resluts.txt", "a") as myfile:
# #     #     myfile.write("Accuracy in Training Set: " + "\n")
# #     #     myfile.write(str(Results_Train) + '\n')
# #     #     myfile.write("Accuracy in Validation Set: " + "\n")
# #     #     myfile.write(str(Results_Validation) + '\n')
# #     #     myfile.write("Accuracy in Test Set: " + "\n")
# #     #     myfile.write(str(Results_test) + '\n')
    

    