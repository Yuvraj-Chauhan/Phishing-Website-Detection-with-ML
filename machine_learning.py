# Step 1 import libraries
import pandas as pd
import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt


# Step 2 read the csv files and create pandas dataframes
legitimate_df = pd.read_csv("datasets//structured_data_legitimate.csv")
phishing_df = pd.read_csv("datasets//structured_data_phishing.csv")


# Step 3 combine legitimate and phishing dataframes, and shuffle
df_phish = pd.DataFrame(phishing_df)
df_legit = pd.DataFrame(legitimate_df)

df_phish_dropped = df_phish.drop_duplicates()
# print(df_phish_dropped.shape)

df_legit_dropped = df_legit.drop_duplicates()
# print(df_legit_dropped.shape)

df_concat = pd.concat([df_legit_dropped, df_phish_dropped], axis=0)
# print(df_concat.shape)


# Step 4 remove'url' and remove duplicates, then we can create X and Y for the models, Supervised Learning
df_concat_dropped = df_concat.drop_duplicates()
# print(df_concat_dropped.shape)

df = df_concat_dropped.sample(frac=1)


df = df.drop('URL', axis=1)

X = df.drop(columns=['label', 'has_audio', 'has_text_area', 'has_email_input', 'number_of_images', 'number_of_tr', 'number_of_table'], axis=1)

Y = df['label']


# Step 5 split data to train and test
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)


# Step 6 create a ML model using sklearn
# Random Forest
rf_model = RandomForestClassifier(n_estimators=150, random_state=5)

# Decision Tree
dt_model = tree.DecisionTreeClassifier()

# AdaBoost
ab_model = AdaBoostClassifier()

# SVM 
# svm_model = svm.LinearSVC()

# Gaussian Naive Bayes
nb_model = GaussianNB()

# Neural Network
nn_model = MLPClassifier(alpha=1)

# KNNeighborsClassifier
knn_model = KNeighborsClassifier()

# Step 7 train the model
# rf_model.fit(x_train, y_train)


# Step 8 make some predictions using test data
# predictions = rf_model.predict(x_test)


# Step 9 create a confusion matrix and tn, tp, fn , fp
# tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictions).ravel()


# Step 10 calculate accuracy, precision and recall scores
# accuracy = (tp + tn) / (tp + tn + fp + fn)
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)

# print("accuracy --> ", accuracy)
# print("precision --> ", precision)
# print("recall --> ", recall)


# K-fold cross validation with K = 5
K = 5
total = X.shape[0]
index = int(total / K)

# 1
X_1_test = X.iloc[:index]
X_1_train = X.iloc[index:]
Y_1_test = Y.iloc[:index]
Y_1_train = Y.iloc[index:]

# 2
X_2_test = X.iloc[index:index*2]
X_2_train = X.iloc[np.r_[:index, index*2:]]
Y_2_test = Y.iloc[index:index*2]
Y_2_train = Y.iloc[np.r_[:index, index*2:]]

# 3
X_3_test = X.iloc[index*2:index*3]
X_3_train = X.iloc[np.r_[:index*2, index*3:]]
Y_3_test = Y.iloc[index*2:index*3]
Y_3_train = Y.iloc[np.r_[:index*2, index*3:]]

# 4
X_4_test = X.iloc[index*3:index*4]
X_4_train = X.iloc[np.r_[:index*3, index*4:]]
Y_4_test = Y.iloc[index*3:index*4]
Y_4_train = Y.iloc[np.r_[:index*3, index*4:]]

# 5
X_5_test = X.iloc[index*4:]
X_5_train = X.iloc[:index*4]
Y_5_test = Y.iloc[index*4:]
Y_5_train = Y.iloc[:index*4]


# X and Y train and test lists
X_train_list = [X_1_train, X_2_train, X_3_train, X_4_train, X_5_train]
X_test_list = [X_1_test, X_2_test, X_3_test, X_4_test, X_5_test]

Y_train_list = [Y_1_train, Y_2_train, Y_3_train, Y_4_train, Y_5_train]
Y_test_list = [Y_1_test, Y_2_test, Y_3_test, Y_4_test, Y_5_test]


def calculate_measures(TN, TP, FN, FP):
    model_accuracy = (TP + TN) / (TP + TN + FN + FP)
    model_precision = TP / (TP + FP)
    model_recall = TP / (TP + FN)
    return model_accuracy, model_precision, model_recall


rf_accuracy_list, rf_precision_list, rf_recall_list, rf_f1_list = [], [], [], []
dt_accuracy_list, dt_precision_list, dt_recall_list, dt_f1_list = [], [], [], []
ab_accuracy_list, ab_precision_list, ab_recall_list, ab_f1_list = [], [], [], []
# svm_accuracy_list, svm_precision_list, svm_recall_list, rf_f1_list = [], [], [], []
nb_accuracy_list, nb_precision_list, nb_recall_list, nb_f1_list = [], [], [], []
nn_accuracy_list, nn_precision_list, nn_recall_list, nn_f1_list = [], [], [], []
knn_accuracy_list, knn_precision_list, knn_recall_list, knn_f1_list = [], [], [], []


for i in range(0, K):
    # --- RANDOM FOREST ---
    rf_model.fit(X_train_list[i], Y_train_list[i])
    rf_predictions = rf_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=rf_predictions).ravel()
    rf_accuracy, rf_precision, rf_recall = calculate_measures(tn, tp, fn, fp)
    rf_accuracy_list.append(rf_accuracy)
    rf_precision_list.append(rf_precision)
    rf_recall_list.append(rf_recall)
    rf_f1_list.append(f1_score(y_true=Y_test_list[i], y_pred=rf_predictions))

    # --- DECISION TREE ---
    dt_model.fit(X_train_list[i], Y_train_list[i])
    dt_predictions = dt_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=dt_predictions).ravel()
    dt_accuracy, dt_precision, dt_recall = calculate_measures(tn, tp, fn, fp)
    dt_accuracy_list.append(dt_accuracy)
    dt_precision_list.append(dt_precision)
    dt_recall_list.append(dt_recall)
    dt_f1_list.append(f1_score(y_true=Y_test_list[i], y_pred=dt_predictions))

    # --- SUPPORT VECTOR MACHINE ---
    # svm_model.fit(X_train_list[i], Y_train_list[i])
    # svm_predictions = svm_model.predict(X_test_list[i])
    # tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=svm_predictions).ravel()
    # svm_accuracy, svm_precision, svm_recall = calculate_measures(tn, tp, fn, fp)
    # svm_accuracy_list.append(svm_accuracy)
    # svm_precision_list.append(svm_precision)
    # svm_recall_list.append(svm_recall)
    # svm_f1_list.append(f1_score(y_true=Y_test_list[i], y_pred=svm_predictions))

    # --- ADABOOST ---
    ab_model.fit(X_train_list[i], Y_train_list[i])
    ab_predictions = ab_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=ab_predictions).ravel()
    ab_accuracy, ab_precision, ab_recall = calculate_measures(tn, tp, fn, fp)
    ab_accuracy_list.append(ab_accuracy)
    ab_precision_list.append(ab_precision)
    ab_recall_list.append(ab_recall)
    ab_f1_list.append(f1_score(y_true=Y_test_list[i], y_pred=ab_predictions))

    # --- GAUSSIAN NAIVE BAYES --- 
    nb_model.fit(X_train_list[i], Y_train_list[i])
    nb_predictions = nb_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=nb_predictions).ravel()
    nb_accuracy, nb_precision, nb_recall = calculate_measures(tn, tp, fn, fp)
    nb_accuracy_list.append(nb_accuracy)
    nb_precision_list.append(nb_precision)
    nb_recall_list.append(nb_recall)
    nb_f1_list.append(f1_score(y_true=Y_test_list[i], y_pred=nb_predictions))

    # --- NEURAL NETWORK ---
    nn_model.fit(X_train_list[i], Y_train_list[i])
    nn_predictions = nn_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=nn_predictions).ravel()
    nn_accuracy, nn_precision, nn_recall = calculate_measures(tn, tp, fn, fp)
    nn_accuracy_list.append(nn_accuracy)
    nn_precision_list.append(nn_precision)
    nn_recall_list.append(nn_recall)
    nn_f1_list.append(f1_score(y_true=Y_test_list[i], y_pred=nn_predictions))

    # --- K-NEIGHBOURS CLASSIFIER ---
    knn_model.fit(X_train_list[i], Y_train_list[i])
    knn_predictions = knn_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=knn_predictions).ravel()
    knn_accuracy, knn_precision, knn_recall = calculate_measures(tn, tp, fn, fp)
    knn_accuracy_list.append(knn_accuracy)
    knn_precision_list.append(knn_precision)
    knn_recall_list.append(knn_recall)
    knn_f1_list.append(f1_score(y_true=Y_test_list[i], y_pred=knn_predictions))
    

RF_accuracy = sum(rf_accuracy_list) / len(rf_accuracy_list)
RF_precision = sum(rf_precision_list) / len(rf_precision_list)
RF_recall = sum(rf_recall_list) / len(rf_recall_list)
RF_f1 = sum(rf_f1_list)/ len(rf_f1_list)
# print("Random Forest accuracy ==> ", RF_accuracy)
# print("Random Forest precision ==> ", RF_precision)
# print("Random Forest recall ==> ", RF_recall)


DT_accuracy = sum(dt_accuracy_list) / len(dt_accuracy_list)
DT_precision = sum(dt_precision_list) / len(dt_precision_list)
DT_recall = sum(dt_recall_list) / len(dt_recall_list)
DT_f1 = sum(dt_f1_list)/ len(dt_f1_list)

# print("Decision Tree accuracy ==> ", DT_accuracy)
# print("Decision Tree precision ==> ", DT_precision)
# print("Decision Tree recall ==> ", DT_recall)


AB_accuracy = sum(ab_accuracy_list) / len(ab_accuracy_list)
AB_precision = sum(ab_precision_list) / len(ab_precision_list)
AB_recall = sum(ab_recall_list) / len(ab_recall_list)
AB_f1 = sum(ab_f1_list)/ len(ab_f1_list)


# print("AdaBoost accuracy ==> ", AB_accuracy)
# print("AdaBoost precision ==> ", AB_precision)
# print("AdaBoost recall ==> ", AB_recall)


# SVM_accuracy = sum(svm_accuracy_list) / len(svm_accuracy_list)
# SVM_precision = sum(svm_precision_list) / len(svm_precision_list)
# SVM_recall = sum(svm_recall_list) / len(svm_recall_list)
# SVM_f1 = sum(svm_f1_list)/ len(svm_f1_list)


# print("Support Vector Machine accuracy ==> ", SVM_accuracy)
# print("Support Vector Machine precision ==> ", SVM_precision)
# print("Support Vector Machine recall ==> ", SVM_recall)


NB_accuracy = sum(nb_accuracy_list) / len(nb_accuracy_list)
NB_precision = sum(nb_precision_list) / len(nb_precision_list)
NB_recall = sum(nb_recall_list) / len(nb_recall_list)
NB_f1 = sum(nb_f1_list)/ len(nb_f1_list)

# print("Gaussian Naive Bayes accuracy ==> ", NB_accuracy)
# print("Gaussian Naive Bayes precision ==> ", NB_precision)
# print("Gaussian Naive Bayes recall ==> ", NB_recall)


NN_accuracy = sum(nn_accuracy_list) / len(nn_accuracy_list)
NN_precision = sum(nn_precision_list) / len(nn_precision_list)
NN_recall = sum(nn_recall_list) / len(nn_recall_list)
NN_f1 = sum(nn_f1_list)/ len(nn_f1_list)

# print("Neural Network accuracy ==> ", NN_accuracy)
# print("Neural Network precision ==> ", NN_precision)
# print("Neural Network recall ==> ", NN_recall)


KNN_accuracy = sum(knn_accuracy_list) / len(knn_accuracy_list)
KNN_precision = sum(knn_precision_list) / len(knn_precision_list)
KNN_recall = sum(knn_recall_list) / len(knn_recall_list)
KNN_f1 = sum(knn_f1_list)/ len(knn_f1_list)

# print("K-Neighbours Classifier accuracy ==> ", KNN_accuracy)
# print("K-Neighbours Classifier precision ==> ", KNN_precision)
# print("K-Neighbours Classifier recall ==> ", KNN_recall)



data = {
    'accuracy': [NB_accuracy, DT_accuracy, RF_accuracy, AB_accuracy, NN_accuracy, KNN_accuracy],

    'precision': [NB_precision, DT_precision, RF_precision, AB_precision, NN_precision, KNN_precision],

    'recall': [NB_recall, DT_recall, RF_recall, AB_recall, NN_recall, KNN_recall],

    'f1 Score': [NB_f1, DT_f1, RF_f1, AB_f1, NN_f1, KNN_f1]
        }

index = ['NB', 'DT', 'RF', 'AB', 'NN', 'KNN']

df_results = pd.DataFrame(data=data, index=index)
print(df_results.head())

'''
# Plotting the Dataframe
ax = df_results.plot.bar(rot=0)
plt.show()
'''

'''
Results:
    accuracy   precision   recall   f1 Score
NB  0.843418   0.770461  0.969764   0.858669
DT  0.973196   0.963874  0.982245   0.972928 RANK2
RF  0.981614   0.981504  0.980678   0.981052 RANK1
AB  0.913608   0.898990  0.928133   0.913323
NN  0.903386   0.886326  0.923013   0.903638
'''