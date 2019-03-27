print()

#import packages
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score

#load the dataset
heart_disease=pd.read_csv('desktop/heart.csv')
heart_disease.head(10)
heart_disease.describe()
heart_disease.isnull().sum()#no missing value, cool!!!
heart_disease.columns
#sex: 1 for male and 0 for female
#cp:4 chest pain types
#trestbps:resting blood pressure
#chol:serum cholestoral level
#fbs:if fasting blood sugar > 120mg/dl (0 and 1)
#restecg:resting electrocardiographic results
#thalach:max. heart rate achieved
#exang:exercise induced angina (1=yes; 0=no)
#oldpeak:ST depression induced by exercise relative to rest
#slope:the slope of the peak exercise ST segment
#ca:number of major vessels (0-3) colored by flourosopy
#thal:a blood disease called thalassemmia,
#     3 = normal; 6 = fixed defect; 7 = reversable defect

#alter column names
heart_disease.columns=['age', 'sex', 'chest_pain_type',
                       'resting_blood_pressure', 'cholesterol',
                       'fasting_blood_sugar','rest_ecg',
                       'max_heart_rate_achived', 'exercise_induced_angina',
                       'st_depression', 'st_slope', 'num_major_vessels',
                       'thalassemia', 'target']
heart_disease.info()

#diagnosed heart disease or not
target_count_dist=heart_disease['target'].value_counts()
target_count_dist.plot(kind='bar')

#gender distribution
male=len(heart_disease[heart_disease['sex']==1])
female=len(heart_disease[heart_disease['sex']==0])
#plot
labels='Male', 'Female'
sizes=male, female
colors=['skyblue', 'yellowgreen']
explode=[0.1, 0]
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.show()

plt.figure(figsize=(15,6))
sns.countplot(x='sex',data = heart_disease, hue = 'target',palette='GnBu')
plt.show()

#chest pain type
plt.figure(figsize=(8,6))

label_1='Typical Angina', 'Atypical Angina', 'Nonanginal Pain', 'Asymptomatic'
heart_disease['chest_pain_type'].unique()
size_1=[len(heart_disease[heart_disease['chest_pain_type']==0]),
        len(heart_disease[heart_disease['chest_pain_type']==1]),
        len(heart_disease[heart_disease['chest_pain_type']==2]),
        len(heart_disease[heart_disease['chest_pain_type']==3])]
colors=['skyblue', 'yellowgreen', 'orange', 'purple']
explode=[0,0,0,0]
plt.pie(size_1, explode=explode, labels=label_1, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.show()

plt.figure(figsize=(15,6))
sns.countplot(x='chest_pain_type',data = heart_disease, hue = 'target',palette='GnBu')
plt.show()

#fasting blood sugar
plt.figure(figsize=(8,6))

label_2='Fasting Blood Sugar<120mg/dl', 'Fasting Blood Sugar>120mg/dl'
size_2=[len(heart_disease[heart_disease['fasting_blood_sugar']==0]),
        len(heart_disease[heart_disease['fasting_blood_sugar']==1])]
explode=[0.1,0]
plt.pie(size_2, explode=explode, labels=label_2, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=180)

#exercised-induced angina
plt.figure(figsize=(8,6))

label_3='Yes', 'No'
size_3=[len(heart_disease[heart_disease['exercise_induced_angina']==1]),
        len(heart_disease[heart_disease['exercise_induced_angina']==0])]
explode=[0.1,0]
plt.pie(size_3, explode=explode, labels=label_3, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=180)

#age group
heart_disease['age'].unique()

def age_group(age):
    if age < 35:
        return 'under 35'
    if age >= 35 and age < 45:
        return ' 35-45'
    if age >= 45 and age < 55:
        return '45-55'
    if age >= 55 and age < 65:
        return '55-65'
    if age >= 65 and age < 75:
        return '65-75'
    else:
        return 'over 75'
    
heart_disease['age_group']=heart_disease.age.apply(age_group)
plt.figure(figsize=(15,6))
sns.countplot(x='age_group',data = heart_disease, hue = 'target',palette='GnBu')
plt.show()
#age between 55 and 65 is the largest group

sns.set_style('whitegrid')
#correlation
plt.figure(figsize=(14,8))
sns.heatmap(heart_disease.corr(), annot = True, cmap='coolwarm',linewidths=.1)
plt.show()


#split data
x_train, x_test, y_train, y_test=train_test_split(heart_disease.drop('target', 1),
                                                  heart_disease['target'],test_size=.3,
                                                  random_state=20)

#scaling features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

#Logistic Regression
logit=LogisticRegression()
params={'penalty':['l1', 'l2'],
        'C':[0.01, 0.1, 1, 10, 100],
        'class_weight':['balanced', None]}
log_model = GridSearchCV(logit,param_grid=params,cv=10)
log_model.fit(x_train, y_train)
log_model.best_params_
#L2 penalty, 0.01, banlanced class weight

pred_1 = log_model.predict(x_test)
print('Accuracy Score: ',accuracy_score(y_test,pred_1))
print('Using Logistic Regression we get an accuracy score of: ',
      round(accuracy_score(y_test,pred_1),5)*100,'%')

print(classification_report(y_test,pred_1))
#                  precision  recall   f1-score   support

#           0       0.73      0.82      0.77        39
#           1       0.85      0.77      0.81        52

#   micro avg       0.79      0.79      0.79        91
#   macro avg       0.79      0.79      0.79        91
#weighted avg       0.80      0.79      0.79        91

#confusion matrix
cnf_1=confusion_matrix(y_test, pred_1)
cnf_1

#Specificity and sensitivity
total_1=sum(sum(cnf_1))
sensitivity_1=cnf_1[0,0]/(cnf_1[0,0] + cnf_1[1,0])
specificity_1=cnf_1[1,1]/(cnf_1[1,1] + cnf_1[0,1])
print('Sensitivity: ', sensitivity_1)
print('Specificity: ', specificity_1)
#sensitivity: 0.72, specificity: 0.85

#create the heat map
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

sns.heatmap(pd.DataFrame(cnf_1), annot = True, cmap = 'YlGnBu',
           fmt = 'g')
plt.tight_layout()
plt.title('Confusion matrix for Logistic Regression Model', y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#ROC curve
target_probailities_log = log_model.predict_proba(x_test)[:,1]
log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,
                                                             target_probailities_log)
plt.figure(figsize=(10,6))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(log_false_positive_rate,log_true_positive_rate)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()

#Decision Tree
DT= DecisionTreeClassifier(random_state=7)
parameter_1 = {'max_features': ['auto', 'sqrt', 'log2'],
               'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
               'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}
tree_model = GridSearchCV(DT, param_grid=parameter_1, n_jobs=-1)
tree_model.fit(x_train, y_train)
tree_model.best_params_
#max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 14

pred_2=tree_model.predict(x_test)
print('Accuracy Score: ',accuracy_score(y_test,pred_2))
print('Using Decision Tree we get an accuracy score of: ',
      round(accuracy_score(y_test,pred_2),5)*100,'%')

print(classification_report(y_test,pred_2))
#                 precision  recall   f1-score    support

#           0       0.76      0.72      0.74        39
#           1       0.80      0.83      0.81        52

#   micro avg       0.78      0.78      0.78        91
#   macro avg       0.78      0.77      0.77        91
#weighted avg       0.78      0.78      0.78        91

#confusion matrix
cnf_2=confusion_matrix(y_test, pred_2)
cnf_2

total_1=sum(sum(cnf_2))
sensitivity_2=cnf_2[0,0]/(cnf_2[0,0] + cnf_2[1,0])
specificity_2=cnf_2[1,1]/(cnf_2[1,1] + cnf_2[0,1])
print('Sensitivity: ', sensitivity_2)
print('Specificity: ', specificity_2)
#sensitivity:0.76, specificity:0.80

class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

sns.heatmap(pd.DataFrame(cnf_2), annot = True, cmap = 'YlGnBu',
           fmt = 'g')
plt.tight_layout()
plt.title('Confusion matrix for Decision Tree Model', y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#ROC curve
target_probailities_tree = tree_model.predict_proba(x_test)[:,1]
tree_false_positive_rate,tree_true_positive_rate,tree_threshold = roc_curve(y_test,
                                                             target_probailities_tree)

plt.figure(figsize=(10,6))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(tree_false_positive_rate,tree_true_positive_rate)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()

#Random Forest
RF=RandomForestClassifier(random_state=1)
from pprint import pprint
print('Parameters currently in use:\n')
pprint(RF.get_params())

parameter_2={'n_estimators':[200,300,500],
             'max_features':['auto', 'sqrt'],
             'max_depth':[int(i) for i in np.linspace(10, 20, num=2)],
             'min_samples_split':[2], 'min_samples_leaf':[2,4,8],
             'bootstrap':[True]}
#After tuning hyperparameters, these coefficients are better-fit.
RF_model = GridSearchCV(RF, param_grid=parameter_2, n_jobs=-1)
RF_model.fit(x_train, y_train)
RF_model.best_params_
#bootstrap': True,'max_depth': 5,'max_features': 'auto','min_samples_leaf': 16,
#'min_samples_split': 2,'n_estimators': 75

pred_3=RF_model.predict(x_test)

print('Accuracy Score: ',accuracy_score(y_test,pred_3))
print('Using Random Forest we get an accuracy score of: ',
      round(accuracy_score(y_test,pred_3),5)*100,'%')

print(classification_report(y_test,pred_3))
#                 precision  recall   f1-score   support
#
#           0       0.77      0.85      0.80        39
#           1       0.88      0.81      0.84        52

#   micro avg       0.82      0.82      0.82        91
#   macro avg       0.82      0.83      0.82        91
#weighted avg       0.83      0.82      0.82        91

#confusion matrix
cnf_3=confusion_matrix(y_test, pred_3)
cnf_3

total_3=sum(sum(cnf_3))
sensitivity_3=cnf_3[0,0]/(cnf_3[0,0] + cnf_3[1,0])
specificity_3=cnf_3[1,1]/(cnf_3[1,1] + cnf_3[0,1])
print('Sensitivity: ', sensitivity_3)
print('Specificity: ', specificity_3)
#sensitivity:0.73, specificity:0.85

class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

sns.heatmap(pd.DataFrame(cnf_3), annot = True, cmap = 'YlGnBu',
           fmt = 'g')
plt.tight_layout()
plt.title('Confusion matrix for Random Forest Model', y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#ROC curve
target_probailities_RF = RF_model.predict_proba(x_test)[:,1]
RF_false_positive_rate,RF_true_positive_rate,tree_threshold = roc_curve(y_test,
                                                             target_probailities_RF)

plt.figure(figsize=(10,6))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(RF_false_positive_rate,RF_true_positive_rate)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()

#Compare accuracy among 3 models
print('Accuracy score for Logistic Regression:', round(accuracy_score(y_test,pred_1),5)*100, '%',
      '\n', 'Accuracy score for Decission Tree:', round(accuracy_score(y_test,pred_2),5)*100, '%',
      '\n', 'Accuracy score for Random Forest:', round(accuracy_score(y_test,pred_3),5)*100, '%')
#Accuracy for 3 algorithms:
#Logistic regression: 79.12%; Decision Tree: 78.02%; Random Forest: 80.22%

#Plot ROC curve
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(log_false_positive_rate,log_true_positive_rate,label='Logistic Regression')
plt.plot(tree_false_positive_rate,tree_true_positive_rate,label='Decision Tree')
plt.plot(RF_false_positive_rate,RF_true_positive_rate,label='Random Forest')
plt.plot([0,1], ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.legend()
plt.show()
#Random Forest has higher accuracy than other 2 models.