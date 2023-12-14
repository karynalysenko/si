import os
import sys
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
# sys.path.append(PARENT_DIR)
# print(PARENT_DIR)

#export': export PYTHONPATH="${PYTHONPATH}:/home/karyna/Documents/SIB/si/src/si/"


from src.io_.csv_file import read_csv

filename= f'{PARENT_DIR}/datasets/iris/iris.csv'
# filename= f'{PARENT_DIR}/datasets/iris/iris_missing_data.csv'

##exercise 1
##1.1
iris_dataset = read_csv(filename, features = True, label= True)
print(iris_dataset.X) #Resposta



# #1.2
penultimo_feature_name=iris_dataset.features[-2]
# print(penultimo_feature_name)

penultimo_column=iris_dataset.X[:,-2]
print(penultimo_column.shape) #Resposta

# #1.3
last_samples = iris_dataset.X[-10:]
l,c = last_samples.shape
for x in range(0,c):
    a = np.mean(last_samples[:,x])
    print(iris_dataset.features[x], a) #Resposta

# #1.4
value_filtered=len(iris_dataset.X[iris_dataset.X<=6])
print(value_filtered) #Resposta

# #1.5
filtered=iris_dataset.X[iris_dataset.label!="Iris-setosa"]
print(filtered.size) #Resposta


# #exercicio 2 testes (Optional) -> novos metodos em si/data/dataset
##2.1 
# # print(iris_dataset.X[~np.isnan(iris_dataset.X)]) #filtrar os nan
a=iris_dataset.dropna()
# print(a.X)
x=iris_dataset.fillna(9.9)
# print(x.X)
y=iris_dataset.remove_by_index(1)
# print(y.X)

# #3.1
from src.si.feature_selection.select_percentile import SelectPercentile
# #3.2 testes
dataset = read_csv(filename, features = True, label= True)
selector = SelectPercentile(percentile=50)
selector = selector.fit(dataset)
# print(selector.p) #p-value para cada feature estimada por score_func
dataset = selector.transform(dataset)
print(dataset.features) #Resposta


#4 
#implementaçao de metodos
from src.si.statistics import manhattan_distance

##5.1
##implementaçao de metodos
from src.si.decomposition.pca import PCA
# #5.2
filename= f'{PARENT_DIR}/datasets/iris/iris.csv'
iris_dataset = read_csv(filename, features = True, label= True)

selector = PCA(n_components=2)
selector = selector.fit(iris_dataset)
print("PCA components:\n", selector.n_components)
print("Explained variance:\n", selector.explained_variance)
selector = selector.fit_transform(iris_dataset)
print('Dataset trnasformed:', selector)


##6.1
##implementaçao de metodos
from src.si.model_selection.split import stratified_train_test_split
# #6.2
filename= f'{PARENT_DIR}/datasets/iris/iris.csv'
iris_dataset = read_csv(filename, features = True, label= True)
print(stratified_train_test_split(iris_dataset)[0].shape())
print(stratified_train_test_split(iris_dataset)[1].shape())
train_dataset, test_dataset = stratified_train_test_split(iris_dataset)

##7
##7.1
from src.si.metrics import rmse

##7.2 e 7.3
from src.si.models import knn_regressor

##8
from src.si.models import ridge_regression_least_squares

##9
from src.si.models import random_forest_classifier


# #16
# from si.data.dataset import Dataset
# from si.neural_networks.layers import Layer, DenseLayer, Dropout
# from si.neural_networks.activation import ReLUActivation, SigmoidActivation
# from si.neural_networks.losses import  BinaryCrossEntropy
# from si.metrics.accuracy import accuracy
# from io_.csv_file import read_csv
# from src.si.model_selection.split import train_test_split
# from src.si.neural_networks.neural_network import NeuralNetwork
# from src.si.neural_networks.optimizers import SGD

# np.random.seed(42)
# X = np.random.randn(200, 32)  # 160 amostras, 32 características
# y = np.random.randint(2, size=(200, 1))
# dataset = Dataset(X=X,y=y)

# train, test = train_test_split(dataset, test_size=0.3, random_state=42)

# # network
# net = NeuralNetwork(epochs=100, batch_size=16, optimizer=SGD, learning_rate=0.01, verbose=True,
#                     loss=BinaryCrossEntropy, metric=accuracy)
# n_features = dataset.X.shape[1]
# net.add(DenseLayer(16, (n_features,)))
# net.add(ReLUActivation())
# net.add(DenseLayer(8))
# net.add(ReLUActivation())
# net.add(DenseLayer(1))
# net.add(SigmoidActivation())
# net.add(Dropout(0.5))
# # train
# net.fit(train)

# # test
# out = net.predict(test)
# print(net.score(test))

# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras.optimizers import SGD
# from sklearn.model_selection import train_test_split

# model_keras = Sequential()
# model_keras.add(Dense(16, input_shape=(n_features,), activation='relu'))
# model_keras.add(Dense(8, activation='relu'))
# model_keras.add(Dense(1, activation='sigmoid'))

# model_keras.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Treinar o modelo
# history = model_keras.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, validation_data=(X_test, y_test))

# test_loss, test_accuracy = model_keras.evaluate(X_test, y_test)
# print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')