#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Reading the Dataset
dataset = pd.read_csv('data/melb_data.csv')

#Cleaning the Dataset
dataset_rev = dataset.drop(['Address','SellerG', 'Suburb', 'Postcode', 'Method', 'Lattitude', 'Longtitude'], axis=1)

#Taking Price Column to the end of the dataset
price = dataset_rev['Price']
dataset_rev.drop(labels=['Price'], axis=1,inplace = True)
dataset_rev.insert(len(dataset_rev.keys()), 'Price', price)

#Improving Sell Date Relevance
from datetime import datetime
sell_dates = dataset_rev['Date']
for i in range(0, len(sell_dates)):
    time = datetime.strptime(sell_dates[i], '%d/%m/%Y')
    dataset_rev['Date'][i] = (datetime.now() - time).days

#Removing nan values from Building Area and Year Built
dataset_rev['BuildingArea'] = dataset_rev['BuildingArea'].replace(np.nan, 0)
dataset_rev['YearBuilt'] = dataset_rev['YearBuilt'].replace(np.nan, 0)
dataset_rev['Car'] = dataset_rev['Car'].replace(np.nan, 0)
dataset_rev['CouncilArea'] = dataset_rev['CouncilArea'].replace(np.nan, 'None')


X = dataset_rev.iloc[:, :-1].values
y = dataset_rev.iloc[:, -1].values


#Encoding Categorical Data (Region)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct_region = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [11])], remainder="passthrough")
X = np.array(ct_region.fit_transform(X))

#Encoding Categorical Data (Council)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct_council = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [18])], remainder="passthrough")
X = np.array(ct_council.fit_transform(X))


#Encoding Categorical Data (HouseType)
ct_htype = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [43])], remainder="passthrough")
X = np.array(ct_htype.fit_transform(X))


#Splitting Dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


#Training the SVR Model on the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)


#Predicting the Test Results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
y_result = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), axis=1)


#Evaluating Results from Model
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


#Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
print('Accuracy: {:.2f} %'.format(accuracies.mean() * 100))
print('Standard Deviation: {:.2f} %'.format(accuracies.std() * 100))


#Visualising a Test Results
plt.bar(range(0, 100), y_test[0:100], color='blue')
plt.title('Retail Estate Prices (Actual Prices)')
plt.xlabel('Case Number')
plt.ylabel('Sell Price')
plt.show()

#Visualising a Pedicted Results
plt.bar(range(0, 100), y_pred[0:100], color='red')
plt.title('Retail Estate Prices (Predicted)')
plt.xlabel('Case Number')
plt.ylabel('Sell Price')
plt.show()