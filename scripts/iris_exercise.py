import os
import sys
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PARENT_DIR)
# print(PARENT_DIR)

from src.io.csv_file import read_csv

filename= f'{PARENT_DIR}/datasets/iris/iris.csv'
# filename= f'{PARENT_DIR}/datasets/iris/iris_missing_data.csv'

#exercise 1
#1.1
iris_dataset = read_csv(filename, features = True, label= True)
# print(iris_dataset.X)

# print(iris_dataset.X[~np.isnan(iris_dataset.X)])
#1.2
penultimo_feature_name=iris_dataset.features[-2]
# print(penultimo_feature_name)

penultimo_column=iris_dataset.X[:,-2]
# print(penultimo_column.shape)

#1.3
last_samples = iris_dataset.X[-10:]
l,c = last_samples.shape
for x in range(0,c):
    a= np.mean(last_samples[:,x])
    # print(a)

#1.4
value_filtered=len(iris_dataset.X[iris_dataset.X<=6])
# print(value_filtered)

#1.5
filtered=iris_dataset.X[iris_dataset.label!="Iris-setosa"]
# print(filtered.size)


#exercicio 2 testes (Optional)
x=iris_dataset.fillna(9.9)
# print(x.X)
a=iris_dataset.dropna()
# print(a.X)
y=iris_dataset.remove_by_index(1)
# print(y.X)

#3.1
from src.si.feature_selection.select_percentile import SelectPercentile
#3.2 testes
dataset = read_csv(filename, features = True, label= True)
selector = SelectPercentile(percentile=50)
selector = selector.fit(dataset)
print(selector.p)

dataset = selector.transform(dataset)
print(dataset.features)