
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('/mnt/data/train.csv')

# Standardize the star color names
df['Star color'] = df['Star color'].str.lower().str.replace('-', ' ')

# Encode the categorical variables
le = LabelEncoder()
df['Star type'] = le.fit_transform(df['Star type'])
df['Star color'] = le.fit_transform(df['Star color'])
df['Spectral Class'] = le.fit_transform(df['Spectral Class'])

# Separate the features and the target variable
X = df.drop('Star type', axis=1)
y = df['Star type']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42))
]

# Train and evaluate each model
model_names = []
model_scores = []

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_names.append(name)
    model_scores.append(accuracy)

# Create a dataframe to display the model performance
performance_df = pd.DataFrame({'Model': model_names, 'Accuracy': model_scores})

# Display the performance dataframe
print(performance_df)
