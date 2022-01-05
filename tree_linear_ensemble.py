print(__doc__)

# Import the necessary modules and libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt 
import pydotplus
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn import model_selection
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")


# Load data
file = open("/Users/Daniella/Documents/dani_files/Graduate_School/Winter_2019/CSI5155/Project/NFA.csv")
file.readline()
column_names = ['ISO.alpha.3.code','UN_region','UN_subregion','year','record','crop_land','grazing_land','forest_land','fishing_ground','built_up_land','carbon','total','Percapita.GDP..2010.USD.','population']
data = pd.read_csv(file,header=None,names=column_names)

# Feature selection
X = data.drop(labels='carbon',axis=1)
y = data['carbon']

# Initialize y arrays
tree_y_pred=[]
tree_y_test=[]
linear_y_pred=[]
linear_y_test=[]
ensemble_y_pred=[]
ensemble_y_test=[]

# Split data
X_train_array=[]
X_test_array=[]
y_train_array=[]
y_test_array=[]
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    X_train_array.append(X_train)
    X_test_array.append(X_test)
    y_train_array.append(y_train)
    y_test_array.append(y_test)

# Run 10 tree datasets
print("Tree model")
print("Dataset, Mean Absolute Error, Median Absolute Error, R^2 Regression Score")
for i in range(10):
    
    # Build tree model
    regr = DecisionTreeRegressor()
    regr.fit(X_train_array[i],y_train_array[i])

    # Predict
    y_pred = regr.predict(X_test_array[i])

    # Add values to array
    tree_y_pred.append(y_pred)
    tree_y_test.append(y_test_array[i])

    # Evaluate model
    # Dataset, Mean Absolute Error, Median Absolute Error, R^2 Regression Score
    print(str(i+1),',', metrics.mean_absolute_error(y_test_array[i], y_pred),',',metrics.median_absolute_error(y_test_array[i], y_pred),',',metrics.r2_score(y_test_array[i], y_pred))  

# Plot tree
plt.Figure()
plt.xlabel('y_test')
plt.ylabel('y_pred')
for i in range(10):
    plt.scatter(tree_y_test[i],tree_y_pred[i], alpha=0.4, edgecolors='w')
    plt.plot(np.unique(tree_y_test[i]), np.poly1d(np.polyfit(tree_y_test[i],tree_y_pred[i],1))(np.unique(tree_y_test[i])), linewidth=0.5)
plt.tight_layout()
plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Tree_y_test_pred_scatter_10.png')
plt.close()

# Run 10 linear datasets
print("Linear model")
print("Dataset, Mean Absolute Error, Median Absolute Error, R^2 Regression Score")
for i in range(10):
    
    # Model
    lm = linear_model.LinearRegression()
    lm.fit(X_train_array[i],y_train_array[i])

    # Predict
    y_pred = lm.predict(X_test_array[i])

    # Add values to array
    linear_y_pred.append(y_pred)
    linear_y_test.append(y_test_array[i])

    # Evaluate
    # Dataset, Mean Absolute Error, Median Absolute Error, R^2 Regression Score
    print(str(i+1),',', metrics.mean_absolute_error(y_test_array[i], y_pred),',',metrics.median_absolute_error(y_test_array[i], y_pred),',',metrics.r2_score(y_test_array[i], y_pred))  

# Plot linear
plt.Figure()
plt.xlabel('y_test')
plt.ylabel('y_pred')
for i in range(10):
    plt.scatter(linear_y_test[i],linear_y_pred[i], alpha=0.4, edgecolors='w')
    plt.plot(np.unique(linear_y_test[i]), np.poly1d(np.polyfit(linear_y_test[i],linear_y_pred[i],1))(np.unique(linear_y_test[i])), linewidth=0.5)
plt.tight_layout()
plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Linear_y_test_pred_scatter_10.png')
plt.close()

# Run 10 ensemble datasets
print("Ensemble model")
print("Dataset, Mean Absolute Error, Median Absolute Error, R^2 Regression Score")
for i in range(10):
    
    # Bagging
    model = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100, random_state=0)
    model.fit(X_train_array[i],y_train_array[i])

    # Predict
    y_pred = model.predict(X_test_array[i])

    # Add values to array
    ensemble_y_pred.append(y_pred)
    ensemble_y_test.append(y_test_array[i])

    # Evaluate model
    # Dataset, Mean Absolute Error, Median Absolute Error, R^2 Regression Score
    print(str(i+1),',', metrics.mean_absolute_error(y_test_array[i], y_pred),',',metrics.median_absolute_error(y_test_array[i], y_pred),',',metrics.r2_score(y_test_array[i], y_pred))  

# Plot
plt.Figure()
plt.xlabel('y_test')
plt.ylabel('y_pred')
for i in range(10):
    plt.scatter(ensemble_y_test[i],ensemble_y_pred[i], alpha=0.4, edgecolors='w')
    plt.plot(np.unique(ensemble_y_test[i]), np.poly1d(np.polyfit(ensemble_y_test[i],ensemble_y_pred[i],1))(np.unique(ensemble_y_test[i])), linewidth=0.5)
plt.tight_layout()
plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Ensemble_y_test_pred_scatter_10.png')
plt.close()
