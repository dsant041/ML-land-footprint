print(__doc__)

# Import the necessary modules and libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt 
import pydotplus


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

# Run 10 datasets
for i in range(10):
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    # Build tree model
    regr = DecisionTreeRegressor()
    tree = regr.fit(X_train,y_train)

    # Predict
    y_pred = regr.predict(X_test)

    # Add values to array
    tree_y_pred.append(y_pred)
    tree_y_test.append(y_test)

    # Evaluate model
    # Dataset, Mean Absolute Error, Median Absolute Error, R^2 Regression Score
    print(str(i+1),',', metrics.mean_absolute_error(y_test, y_pred),',',metrics.median_absolute_error(y_test, y_pred),',',metrics.r2_score(y_test, y_pred))  

# Plot
plt.Figure()
plt.xlabel('y_test')
plt.ylabel('y_pred')
for i in range(10):
    plt.scatter(tree_y_test[i],tree_y_pred[i], alpha=0.4, edgecolors='w')
    plt.plot(np.unique(tree_y_test[i]), np.poly1d(np.polyfit(tree_y_test[i],tree_y_pred[i],1))(np.unique(tree_y_test[i])), linewidth=0.5)
plt.tight_layout()
plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Tree_method/y_test_pred_scatter_10.png')
plt.close()
