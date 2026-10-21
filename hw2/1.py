import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

train_data = pd.read_csv('hw2/titanic/train.csv')
test_data = pd.read_csv('hw2/titanic/test.csv')

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

test_data['Age'] = test_data['Age'].fillna(train_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(train_data['Fare'].median())
test_data['Embarked'] = test_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

sex_map = {'male': 0, 'female': 1}
embarked_map = {'S': 0, 'C': 1, 'Q': 2}

train_data['Sex'] = train_data['Sex'].map(sex_map)
train_data['Embarked'] = train_data['Embarked'].map(embarked_map)

test_data['Sex'] = test_data['Sex'].map(sex_map)
test_data['Embarked'] = test_data['Embarked'].map(embarked_map)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_data[features]
y = train_data['Survived']

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)
print("Accuracy:", accuracy_score(y_valid, y_pred))

tree_rules = export_text(model, feature_names=features)
print("Stablo:\n")
print(tree_rules)

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=features, class_names=['Not Survived', 'Survived'], filled=True)
plt.title("Decision Tree za Titanic podatke")
plt.show()

x_test = test_data[features]
test_predictions = model.predict(x_test)

submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})

print(submission)
