print(__doc__)

# Import the necessary modules and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from orangecontrib.associate.fpgrowth import * 
import Orange
from Orange.data import Domain, DiscreteVariable, ContinuousVariable



# Load data
data = Orange.data.Table.from_file("/Users/Daniella/Documents/dani_files/Graduate_School/Winter_2019/CSI5155/Project/NFAorange.csv")                          

# Transform data to boolean
X, mapping = OneHot.encode(data,include_class=True)

# Fin frequent items
print("100 support")
itemsets_100 = dict(frequent_itemsets(X, 100))
print(len(itemsets_100))
print("200 support")
itemsets_200 = dict(frequent_itemsets(X, 200))
print(len(itemsets_200))
print("500 support")
itemsets_500 = dict(frequent_itemsets(X, 500))
print(len(itemsets_500))
print("1000 support")
itemsets_1000 = dict(frequent_itemsets(X, 1000))
print(len(itemsets_1000))

class_items = {item
               for item, var, _ in OneHot.decode(mapping, data, mapping)
               if var is data.domain.class_var
               }
print(sorted(class_items))

rules = [(P, Q, supp, conf)
             for P, Q, supp, conf in association_rules(itemsets_100, .8)
             if len(Q) == 1 and Q & class_items]
print(len(rules))
print(rules)

# Construct the learner
#rules = Orange.associate.AssociationRulesSparseInducer(data, support=0.3)

# Display rules
#print ("%4s %4s  %s" % ("Supp", "Conf", "Rule"))
#for r in rules[:5]:
#    print ("%4.1f %4.1f  %s" % (r.support, r.confidence, r))
