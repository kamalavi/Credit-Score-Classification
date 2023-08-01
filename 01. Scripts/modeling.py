from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import data_cleaning as dc

random_state = 10

path = r"00. Data\\train.csv"

df, col_float, col_string = dc.data_cleaning(path)

#normalize numeric variables
scaler = MinMaxScaler()
for i in df[col_float]:
    df[i] = scaler.fit_transform(df[[i]])

#creating dummies for categorical variables
for i in col_string:
    df = pd.get_dummies(df, prefix = i, columns = [i], drop_first = False)

#create test, train sets
df_train_x = df.drop(columns='credit_score')
df_train_y = df['credit_score']
x_train, x_test, y_train, y_test = train_test_split(df_train_x,df_train_y,test_size=0.20,random_state=random_state)

obs_map = {1: "Poor", 2: "Standard", 0: "Good"}

#decision tree modelling
dtree = DecisionTreeClassifier(random_state = random_state)
dtree = dtree.fit(x_train, y_train)
y_pred = dtree.predict(x_test)

cm = pd.DataFrame(confusion_matrix(y_test, y_pred)).rename(index = obs_map, columns = obs_map)
accuracy = accuracy_score(y_test, y_pred)
dtree_report = classification_report(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Classification Report \n", dtree_report)

plt.figure()
heatmap = sns.heatmap(cm, annot=True, cmap = "Blues", fmt = 'g')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Decision Tree Model Confusion Matrix HeatMap')
plt.show()                            


#KNN modelling
kn = KNeighborsClassifier(n_neighbors=5,weights = 'distance')
kn.fit(x_train, y_train)
kn_y_pred = kn.predict(x_test)

kn_cm = pd.DataFrame(confusion_matrix(y_test,kn_y_pred)).rename(index = obs_map, columns = obs_map)
kn_accuracy = accuracy_score(y_test,kn_y_pred)
kn_report = classification_report(y_test,kn_y_pred)

print("Accuracy: ", kn_accuracy)
print("Classification Report \n", kn_report)

plt.figure()
heatmap = sns.heatmap(kn_cm, annot=True, cmap = "Blues", fmt = 'g')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('KNN Model Confusion Matrix HeatMap')
plt.show()         

#Random Forest modelling
rf = RandomForestClassifier(n_estimators=100, random_state = random_state)
rf.fit(x_train,y_train)
rf_y_pred = rf.predict(x_test)

rf_cm = pd.DataFrame(confusion_matrix(y_test,rf_y_pred)).rename(index = obs_map, columns = obs_map)
rf_accuracy = accuracy_score(y_test,rf_y_pred)
rf_report = classification_report(y_test,rf_y_pred)

print("Accuracy: ", rf_accuracy)
print("Classification Report \n", rf_report)

plt.figure()
heatmap = sns.heatmap(rf_cm, annot=True, cmap = "Blues", fmt = 'g')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('KNN Model Confusion Matrix HeatMap')
plt.show()         