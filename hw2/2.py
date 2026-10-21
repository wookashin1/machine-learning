import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    for i in range(len(std)):
        if std[i] == 0:
            std[i] = 1
    return (x - mean) / std

def load_data():
    train_data = pd.read_csv('hw2/titanic/train.csv')
    test_data = pd.read_csv('hw2/titanic/test.csv')
    
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
    test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
    train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
    test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
    
    train_data.drop(['Cabin'], axis=1, inplace=True)
    test_data.drop(['Cabin'], axis=1, inplace=True)
    
    train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
    
    if train_data.isnull().any().any() or test_data.isnull().any().any():
        print("GRESKA")
        return None, None, None

    train_data.drop(['Name', 'Ticket'], axis=1, inplace=True)
    test_data.drop(['Name', 'Ticket'], axis=1, inplace=True)

    train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
    test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
    
    train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    test_data['Embarked'] = test_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    x_train = train_data.drop('Survived', axis=1).values
    y_train = train_data['Survived'].values.reshape(-1, 1)
    x_test = test_data.drop('Survived', axis=1, errors='ignore').values
    
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    
    return x_train, y_train, x_test

def initialize_parameters(input_size, hidden_size, output_size):
    w1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    w2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return w1, b1, w2, b2

def forward_propagation(X, w1, b1, w2, b2):
    Z1 = np.dot(X, w1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, w2) + b2
    A2 = sigmoid(Z2)
    return A1, A2

def backward_propagation(X, y, A1, A2, w1, w2, b1, b2, learning_rate):
    m = X.shape[0]
    dZ2 = A2 - y
    dw2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dZ1 = np.dot(dZ2, w2.T) * sigmoid_derivative(A1)
    dw1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    
    return w1, b1, w2, b2

def train(x_train, y_train, input_size, hidden_size, output_size, epochs, learning_rate):
    w1, b1, w2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        A1, A2 = forward_propagation(x_train, w1, b1, w2, b2)
        
        epsilon = 1e-15
        A2 = np.clip(A2, epsilon, 1 - epsilon)

        loss = np.mean(-y_train * np.log(A2 + epsilon) - (1 - y_train) * np.log(1 - A2 + epsilon))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

        w1, b1, w2, b2 = backward_propagation(x_train, y_train, A1, A2, w1, w2, b1, b2, learning_rate)
    
    return w1, b1, w2, b2

def predict(X, w1, b1, w2, b2):
    _, A2 = forward_propagation(X, w1, b1, w2, b2)
    return (A2 > 0.5).astype(int)

x_train, y_train, x_test = load_data()

if x_train is None or y_train is None or x_test is None:
    print("GRESKA")
    exit(0)

split_index = int(0.8 * len(x_train))
x_train_split, X_val_split = x_train[:split_index], x_train[split_index:]
y_train_split, y_val_split = y_train[:split_index], y_train[split_index:]

input_size = x_train.shape[1]
hidden_size = 10
output_size = 1
epochs = 1000
learning_rate = 0.5

w1, b1, w2, b2 = train(x_train_split, y_train_split, input_size, hidden_size, output_size, epochs, learning_rate)

val_predictions = predict(X_val_split, w1, b1, w2, b2)

accuracy = np.mean(val_predictions == y_val_split) * 100
print(f"Accuracy: {accuracy:.2f}%")

predictions = predict(x_test, w1, b1, w2, b2)

print("Predictions:")
