print(__doc__)

# Import the necessary modules and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split


# Load data
file = open("/Users/Daniella/Documents/dani_files/Graduate_School/Winter_2019/CSI5155/Project/NFA.csv")
file.readline()
column_names = ['ISO.alpha.3.code','UN_region','UN_subregion','year','record','crop_land','grazing_land','forest_land','fishing_ground','built_up_land','carbon','total','Percapita.GDP..2010.USD.','population']
data = pd.read_csv(file,header=None,names=column_names)

# Feature selection
X = data.drop(labels='carbon',axis=1)
y = data['carbon']

# Initialize y arrays
linear_y_pred=[]
linear_y_test=[]

# Run 10 datasets
for i in range(10):
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    # Model
    lm = linear_model.LinearRegression()
    lm.fit(X_train,y_train)

    # Predict
    y_pred = lm.predict(X_test)

    # Add values to array
    linear_y_pred.append(y_pred)
    linear_y_test.append(y_test)

    # Evaluate
    # Dataset, Mean Absolute Error, Median Absolute Error, R^2 Regression Score
    print(str(i+1),',', metrics.mean_absolute_error(y_test, y_pred),',',metrics.median_absolute_error(y_test, y_pred),',',metrics.r2_score(y_test, y_pred))  

# Plot
plt.Figure()
plt.xlabel('y_test')
plt.ylabel('y_pred')
for i in range(10):
    plt.scatter(linear_y_test[i],linear_y_pred[i], alpha=0.4, edgecolors='w')
    plt.plot(np.unique(linear_y_test[i]), np.poly1d(np.polyfit(linear_y_test[i],linear_y_pred[i],1))(np.unique(linear_y_test[i])), linewidth=0.5)
plt.tight_layout()
plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Linear/Linear_regr_figs/y_test_pred_scatter_10.png')
plt.close()

