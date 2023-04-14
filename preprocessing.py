import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def min_max_normalization(X):
    min = np.min(X)
    max = np.max(X)
    X = (X - min) / (max - min)
    return X

def z_normalization(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X-mean)/std
    return X


#read CSV file
iris = pd.read_csv('Iris.csv', delimiter=',', header=0, index_col=0)
iris['Species'] = pd.Categorical(iris.Species).codes
#convert categorical data to numeric datairis['Species'] = pd.Categorical(iris.Species).codes

#convert to numpy array
iris = np.array(iris)

#Shuffle data
randNum = np.arange(len(iris))
np.random.shuffle(randNum)
iris=iris[randNum]

X = iris[:, 0:4]
y = iris[:, 4]
print("Please choose your choice to normalize data:")
print("A: No Normalization")
print("B: Min Max Normalization")
print("C: Standard Normalization")
choice = input("You: ")
if choice == "B":
    X = min_max_normalization(X)
elif choice == "C":
    X = z_normalization(X)
elif choice != "A" and choice != "B" and choice != "C":
    print("Invalid Input!")
#Divide data into train and test


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

temp = pd.DataFrame(X_test, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
temp['Species']= y_test
temp.to_csv('Iris_test.csv',index=False)


#save nan file
prob=0.05
for i in range(4):
    nan_mask = np.random.choice([False, True], size=X_train.shape, p=[1-prob, prob])
    print(nan_mask)
    #replace of data points with Nan values
    temp = X_train.copy()
    temp[nan_mask] = np.nan

    #Save the data set with NAN values to a new CSV file
    temp = pd.DataFrame(temp, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    temp['Species']= y_train
    temp.to_csv(f'IrisNan{int(prob * 100)}.csv',index=False)
    prob+= 0.05