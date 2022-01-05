print(__doc__)

# Import the necessary modules and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import model_selection
from sklearn.svm import SVC


# Load data
file = open("/Users/Daniella/Documents/dani_files/Graduate_School/Winter_2019/CSI5155/Project/NFA.csv")
file.readline()
column_names = ['ISO.alpha.3.code','UN_region','UN_subregion','year','record','crop_land','grazing_land','forest_land','fishing_ground','built_up_land','carbon','total','Percapita.GDP..2010.USD.','population']
data = pd.read_csv(file,header=None,names=column_names)

# Feature selection
X = data.drop(labels='carbon',axis=1)
y = data['carbon']

# Initialize y arrays
ensemble_y_pred=[]
ensemble_y_test=[]

# Run 10 datasets
for i in range(10):
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    # Bagging
    model = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100, random_state=0)
    trained_model = model.fit(X_train,y_train)

    # Predict
    y_pred = trained_model.predict(X_test)

    # Add values to array
    ensemble_y_pred.append(y_pred)
    ensemble_y_test.append(y_test)

    # Evaluate model
    # Dataset, Mean Absolute Error, Median Absolute Error, R^2 Regression Score
    print(str(i+1),',', metrics.mean_absolute_error(y_test, y_pred),',',metrics.median_absolute_error(y_test, y_pred),',',metrics.r2_score(y_test, y_pred))  

# Plot
plt.Figure()
plt.xlabel('y_test')
plt.ylabel('y_pred')
for i in range(10):
    plt.scatter(ensemble_y_test[i],ensemble_y_pred[i], alpha=0.4, edgecolors='w')
    plt.plot(np.unique(ensemble_y_test[i]), np.poly1d(np.polyfit(ensemble_y_test[i],ensemble_y_pred[i],1))(np.unique(ensemble_y_test[i])), linewidth=0.5)
plt.tight_layout()
plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Ensemble/Ensemble_figs/y_test_pred_scatter_10.png')
plt.close()
