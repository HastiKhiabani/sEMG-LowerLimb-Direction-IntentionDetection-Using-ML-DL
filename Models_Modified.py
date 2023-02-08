#import HastiGetDatafromPythonScript
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
#from matplotlib.ticker import NullFormatter 
#import matplotlib.ticker as ticker
#from sklearn import preprocessing
#import itertools
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#%% Models    
def train_model_KNN(X_train, y_train, TrainedFolderName):
    tuned_parameters = {'n_neighbors':list(range(2,20,2)),
              'algorithm':['auto'],
              'n_jobs':[-1]} #'n_jobs':None[-1]
    
    knn = KNeighborsClassifier() 
    gsc = GridSearchCV(estimator=knn,
                       param_grid=tuned_parameters, scoring='accuracy', cv=10)
    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_
    ValSet_accuracy= grid_result.best_score_
    results = pd.DataFrame(grid_result.cv_results_)
    
    print("Best hyperparameteres knn: ", best_params)
    print("Validation set Accuracy_Knn: ", ValSet_accuracy)

    with open("%s/StepbyStepResults.txt" %TrainedFolderName, "a") as myfile:
        myfile.write("Best hyperparameteres Knn: " + "\n")
        myfile.write(str(best_params) + '\n')
        myfile.write("ValSet_accuracy Knn: " + "\n")
        myfile.write(str(ValSet_accuracy) + '\n') 

    results.to_excel(TrainedFolderName + '/Knn_resultsoutput.xlsx', sheet_name='Sheet_name_1') 
         
    # Its important to use binary mode 
    knnPickle = open(TrainedFolderName+'/Knnpickle_file', 'wb') 
    # source, destination 
    pickle.dump(grid_result, knnPickle)   #->pickle.dump(model, open(filename, 'wb'))
        
    return grid_result, ValSet_accuracy, results

#%%
def train_model_LDA(X_train, y_train, TrainedFolderName): 
    tuned_parameters = {'solver':['svd', 'lsqr']}

    Lda = LDA() 
    gsc = GridSearchCV(estimator=Lda,
                       param_grid=tuned_parameters, scoring='accuracy', cv=10)
    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_
    ValSet_accuracy= grid_result.best_score_
    results = pd.DataFrame(grid_result.cv_results_)
    
    print("Best hyperparameteres Lda: ", best_params)
    print("Validation set Accuracy_Lda: ", ValSet_accuracy)

    with open("%s/StepbyStepResults.txt" %TrainedFolderName, "a") as myfile:
        myfile.write("Best hyperparameteres Lda: " + "\n")
        myfile.write(str(best_params) + '\n')
        myfile.write("ValSet_accuracy Lda: " + "\n")
        myfile.write(str(ValSet_accuracy) + '\n')        

    results.to_excel(TrainedFolderName + '/Lda_resultsoutput.xlsx', sheet_name='Sheet_name_1') 
        
    # Its important to use binary mode 
    LdaPickle = open(TrainedFolderName+'/Ldapickle_file', 'wb') 
    # source, destination 
    pickle.dump(grid_result, LdaPickle)   #->pickle.dump(model, open(filename, 'wb'))
    
    return grid_result, ValSet_accuracy, results
    
#%%
def train_model_QDA(X_train, y_train, TrainedFolderName): 
    tuned_parameters = params = [{'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]}]
    
    Qda = QDA() 
    gsc = GridSearchCV(estimator=Qda,
                       param_grid=tuned_parameters, scoring='accuracy', cv=10)
    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_
    ValSet_accuracy= grid_result.best_score_
    results = pd.DataFrame(grid_result.cv_results_)
    
    print("Best hyperparameteres Qda: ", best_params)
    print("Validation set Accuracy_Qda: ", ValSet_accuracy)

    with open("%s/StepbyStepResults.txt" %TrainedFolderName, "a") as myfile:
        myfile.write("Best hyperparameteres Qda: " + "\n")
        myfile.write(str(best_params) + '\n')
        myfile.write("ValSet_accuracy Qda: " + "\n")
        myfile.write(str(ValSet_accuracy) + '\n') 

    results.to_excel(TrainedFolderName + '/Qda_resultsoutput.xlsx', sheet_name='Sheet_name_1') 
        
    # Its important to use binary mode 
    QdaPickle = open(TrainedFolderName+'/Qdapickle_file', 'wb') 
    # source, destination 
    pickle.dump(grid_result, QdaPickle)   #->pickle.dump(model, open(filename, 'wb'))
    
    return grid_result, ValSet_accuracy, results    
#%%
def train_model_SVM(X_train, y_train, TrainedFolderName):
    # Set the parameters by cross-validation
    tuned_parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.1, 1, 10]} 
    
    svc = SVC() 
    gsc = GridSearchCV(estimator=svc,
                       param_grid=tuned_parameters, scoring='accuracy', cv=10)
    # SVM.fit(X_train, y_train)
    # yhat = SVM.predict(X_validation)
    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_
    ValSet_accuracy= grid_result.best_score_
    results = pd.DataFrame(grid_result.cv_results_)
    
    print("Best hyperparameteres SVM: ", best_params)
    print("Validation set Accuracy_SVM: ", ValSet_accuracy)

    with open("%s/StepbyStepResults.txt" %TrainedFolderName, "a") as myfile:
        myfile.write("Best hyperparameteres SVM: " + "\n")
        myfile.write(str(best_params) + '\n')
        myfile.write("ValSet_accuracy SVM: " + "\n")
        myfile.write(str(ValSet_accuracy) + '\n') 

    results.to_excel(TrainedFolderName + '/SVM_resultsoutput.xlsx', sheet_name='Sheet_name_1') 
        
    # Its important to use binary mode 
    svmPickle = open(TrainedFolderName + '/svmpickle_file', 'wb') 
    # source, destination 
    pickle.dump(grid_result, svmPickle)   #->pickle.dump(model, open(filename, 'wb'))
    
    return grid_result, ValSet_accuracy, results  

#%% 
def train_model_DT(X_train, y_train, TrainedFolderName):   
    
    tuned_parameters = {'criterion':['gini','entropy'],
                        'max_depth':np.arange(10, 100, 2)}
    
    DT = DecisionTreeClassifier()  #entropy was better than gini
    gsc = GridSearchCV(estimator=DT,
                       param_grid=tuned_parameters, scoring='accuracy', cv=10)
    
    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_
    ValSet_accuracy= grid_result.best_score_
    results = pd.DataFrame(grid_result.cv_results_)
 
    print("Best hyperparameteres DT: ", best_params)
    print("Validation set Accuracy_DT: ", ValSet_accuracy)

    with open("%s/StepbyStepResults.txt" %TrainedFolderName, "a") as myfile:
        myfile.write("Best hyperparameteres TD: " + "\n")
        myfile.write(str(best_params) + '\n')
        myfile.write("ValSet_accuracy TD: " + "\n")
        myfile.write(str(ValSet_accuracy) + '\n') 

    results.to_excel(TrainedFolderName + '/DT_resultsoutput.xlsx', sheet_name='Sheet_name_1') 
        
    # Its important to use binary mode 
    DTPickle = open(TrainedFolderName + '/DTpickle_file', 'wb') 
    # source, destination 
    pickle.dump(grid_result, DTPickle)   #->pickle.dump(model, open(filename, 'wb'))
        
    return grid_result, ValSet_accuracy, results   

#%% 
def train_model_GNB(X_train, y_train, TrainedFolderName):   
    
    tuned_parameters = {}
        #Train Model and Predict
    GNB = GaussianNB()
    gsc = GridSearchCV(estimator=GNB,
                       param_grid=tuned_parameters, scoring='accuracy', cv=10)
    
    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_
    ValSet_accuracy= grid_result.best_score_
    results = pd.DataFrame(grid_result.cv_results_)
    
    print("Best hyperparameteres GNB: ", best_params)    
    print("Validation set Accuracy_GNB: ", ValSet_accuracy)

    with open("%s/StepbyStepResults.txt" %TrainedFolderName, "a") as myfile:
        myfile.write("Best hyperparameteres GNB: " + "\n")
        myfile.write(str(best_params) + '\n')
        myfile.write("ValSet_accuracy GNB: " + "\n")
        myfile.write(str(ValSet_accuracy) + '\n') 

    results.to_excel(TrainedFolderName + '/GNB_resultsoutput.xlsx', sheet_name='Sheet_name_1') 
        
    # Its important to use binary mode 
    GNBPickle = open(TrainedFolderName+'/GNBpickle_file', 'wb') 
    # source, destination 
    pickle.dump(grid_result, GNBPickle)   #->pickle.dump(model, open(filename, 'wb'))
    
    return grid_result, ValSet_accuracy, results    
    
#%% 
def train_model_RF(X_train, y_train, TrainedFolderName):   

    tuned_parameters= {'n_estimators': np.arange(100, 500, 100)  , 'max_depth' : np.arange(20, 100, 10), 'criterion' :['entropy']}
    #Train Model and Predict
    RF = RandomForestClassifier()   
    gsc = GridSearchCV(estimator=RF,
                       param_grid=tuned_parameters, scoring='accuracy', cv=10)
    
    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_
    ValSet_accuracy= grid_result.best_score_
    results = pd.DataFrame(grid_result.cv_results_)
 
    print("Best hyperparameteres RF: ", best_params)
    print("Validation set Accuracy_RF: ", ValSet_accuracy)

    with open("%s/StepbyStepResults.txt" %TrainedFolderName, "a") as myfile:
        myfile.write("Best hyperparameteres RF: " + "\n")
        myfile.write(str(best_params) + '\n')
        myfile.write("ValSet_accuracy RF: " + "\n")
        myfile.write(str(ValSet_accuracy) + '\n') 

    results.to_excel(TrainedFolderName + '/RF_resultsoutput.xlsx', sheet_name='Sheet_name_1') 
        
    # Its important to use binary mode 
    RFPickle = open(TrainedFolderName+'/RFpickle_file', 'wb') 
    # source, destination 
    pickle.dump(grid_result, RFPickle)   #->pickle.dump(model, open(filename, 'wb'))
        
    return grid_result, ValSet_accuracy, results  
#%%
def train_model_Bag(X_train, y_train, TrainedFolderName):   
    tuned_parameters= {'n_estimators': np.arange(10, 100, 20),
                       'max_features': [0.3, 0.5, 1],
                       'max_samples' : [0.3, 0.5, 1.0]}   # 'bootstrap': [True, False], 'bootstrap_features': [True, False]

    Bag = BaggingClassifier() 
    #Train Model and Predict
    gsc = GridSearchCV(estimator=Bag,
                       param_grid=tuned_parameters, scoring='accuracy', cv=10)
    
    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_
    ValSet_accuracy= grid_result.best_score_
    results = pd.DataFrame(grid_result.cv_results_)

    print("Best hyperparameteres EnsBag: ", best_params)
    print("Validation set Accuracy_EnsBag: ", ValSet_accuracy)

    with open("%s/StepbyStepResults.txt" %TrainedFolderName, "a") as myfile:
        myfile.write("Best hyperparameteres EnsBag: " + "\n")
        myfile.write(str(best_params) + '\n')
        myfile.write("ValSet_accuracy EnsBag: " + "\n")
        myfile.write(str(ValSet_accuracy) + '\n') 

    results.to_excel(TrainedFolderName + '/EnsBag_resultsoutput.xlsx', sheet_name='Sheet_name_1') 
        
    # Its important to use binary mode 
    BagPickle = open(TrainedFolderName+'/Bagpickle_file', 'wb') 
    # source, destination 
    pickle.dump(grid_result, BagPickle)   #->pickle.dump(model, open(filename, 'wb'))
    
    return grid_result, ValSet_accuracy, results  
#%% 
def train_model_Ada(X_train, y_train, TrainedFolderName):   
    #get the best k from RF-Optimizer_model:
    tuned_parameters= {'n_estimators': np.arange(10,100,20),
                       'learning_rate': [0.01, 0.05, 0.1, 1]}
    
    #Train Model and Predict
    Ada = AdaBoostClassifier()   
    gsc = GridSearchCV(estimator=Ada,
                       param_grid=tuned_parameters, scoring='accuracy', cv=10)
    
    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_
    ValSet_accuracy= grid_result.best_score_
    results = pd.DataFrame(grid_result.cv_results_)

    print("Best hyperparameteres EnsAda: ", best_params)
    print("Validation set Accuracy_EnsAda: ", ValSet_accuracy)

    with open("%s/StepbyStepResults.txt" %TrainedFolderName, "a") as myfile:
        myfile.write("Best hyperparameteres EnsAda: " + "\n")
        myfile.write(str(best_params) + '\n')
        myfile.write("ValSet_accuracy EnsAda: " + "\n")
        myfile.write(str(ValSet_accuracy) + '\n') 

    results.to_excel(TrainedFolderName + '/EnsAda_resultsoutput.xlsx', sheet_name='Sheet_name_1') 
        
    # Its important to use binary mode 
    AdaPickle = open(TrainedFolderName+'/Adapickle_file', 'wb') 
    # source, destination 
    pickle.dump(grid_result, AdaPickle)   #->pickle.dump(model, open(filename, 'wb'))
    
    return grid_result, ValSet_accuracy, results 

#%%
def train_model_GBDT(X_train, y_train, TrainedFolderName):   
    #get the best k from GBDT-Optimizer_model:
    tuned_parameters= {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.1],
    "min_samples_split": np.linspace(0.5, 2.5, 3),
    "min_samples_leaf": np.linspace(0.5, 2.5, 3),
    "max_depth":[1,4,7],
    "criterion": ["friedman_mse"],
    "subsample":[0.5, 1.0],
    "n_estimators":[10, 25, 50]
    } #"max_features":["log2","sqrt"],
    #Train Model and Predict
    GBDT = GradientBoostingClassifier() 
    gsc = GridSearchCV(estimator=GBDT,
                       param_grid=tuned_parameters, scoring='accuracy', cv=10)
    
    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_
    ValSet_accuracy= grid_result.best_score_
    results = pd.DataFrame(grid_result.cv_results_)

    print("Best hyperparameteres EnsGBDT: ", best_params)
    print("Validation set Accuracy_EnsGBDT: ", ValSet_accuracy)

    with open("%s/StepbyStepResults.txt" %TrainedFolderName, "a") as myfile:
        myfile.write("Best hyperparameteres GBDT: " + "\n")
        myfile.write(str(best_params) + '\n')
        myfile.write("ValSet_accuracy GBDT: " + "\n")
        myfile.write(str(ValSet_accuracy) + '\n') 

    results.to_excel(TrainedFolderName + '/GBDT_resultsoutput.xlsx', sheet_name='Sheet_name_1') 
        
    # Its important to use binary mode 
    GBDTPickle = open(TrainedFolderName+'/GBDTpickle_file', 'wb') 
    # source, destination 
    pickle.dump(grid_result, GBDTPickle)   #->pickle.dump(model, open(filename, 'wb'))
    
    return grid_result, ValSet_accuracy, results    