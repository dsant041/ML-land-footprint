print(__doc__)

# Import the necessary modules and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import FeatureAgglomeration
import time
import warnings
from sklearn import datasets, mixture
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import cm
from collections import OrderedDict


# Load data
file = open("/Users/Daniella/Documents/dani_files/Graduate_School/Winter_2019/CSI5155/Project/NFA.csv")
file.readline()
column_names = ['ISO.alpha.3.code',
                'UN_region',
                'UN_subregion',
                'year','record',
                'crop_land',
                'grazing_land',
                'forest_land',
                'fishing_ground',
                'built_up_land',
                'carbon','total',
                'Percapita.GDP..2010.USD.',
                'population']
data = pd.read_csv(file,header=None,names=column_names)

geo_f = np.array(data)

# Set up cluster
X = data
#normalize data
X = StandardScaler().fit_transform(data)

# Create clusters
#agg = FeatureAgglomeration(n_clusters=3, linkage='ward')
agg = FeatureAgglomeration(n_clusters=4, linkage='ward')
#agg = FeatureAgglomeration(n_clusters=5, linkage='ward')
agg.fit(X)
reduced = agg.transform(X)
print(agg.labels_)

# Apply clusters to data
X_restored = agg.inverse_transform(reduced)
geo_f_restored = np.reshape(X_restored, geo_f.shape)

print('reduced=',reduced[len(reduced)-1])
print('X_restored=',X_restored[len(X_restored)-1])

# Show plot
plt.Figure()
plt.title('Original data')
sns.heatmap(X, square = False,cmap="Paired")
plt.tight_layout()
#plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Distance_method/Figure_cl_3_original_1.png')
plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Distance_method/Figure_cl_4_original_1.png')
#plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Distance_method/Figure_cl_5_original_1.png')
plt.close()

plt.Figure()
plt.title('Reduced data')
sns.heatmap(reduced, square = False,cmap="Paired")
plt.tight_layout()
#plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Distance_method/Figure_cl_3_reduced_1.png')
plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Distance_method/Figure_cl_4_reduced_1.png')
#plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Distance_method/Figure_cl_5_reduced_1.png')
plt.close()

plt.Figure()
plt.title('Restored data')
sns.heatmap(X_restored, square = False,cmap="Paired")
plt.tight_layout()
#plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Distance_method/Figure_cl_3_restored_1.png')
plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Distance_method/Figure_cl_4_restored_1.png')
#plt.savefig('/Users/Daniella/Documents/Python_files/CSI5155_project/Distance_method/Figure_cl_5_restored_1.png')
plt.close()


