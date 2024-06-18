# nexusfinalproject
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset
diabetes_dataset = pd.read_csv('/content/diabetes (1).csv')
print(diabetes_dataset.head())  # Display the first few rows of the dataset
print(diabetes_dataset.shape)  # Display the shape of the dataset
print(diabetes_dataset.describe())  # Display a statistical summary of the dataset
print(diabetes_dataset['Outcome'].value_counts())  # Display the count of each outcome
print(diabetes_dataset.groupby('Outcome').mean())  # Display the mean values grouped by outcome

# Separate the features and the target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X)  # Display the features
print(Y)  # Display the target

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)  # Display the standardized data

# Update X with the standardized data
X = standardized_data
print(X)  # Display the standardized features

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)  # Display the shapes of the datasets

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Evaluate the classifier on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data:', training_data_accuracy)

# Evaluate the classifier on the testing data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data:', test_data_accuracy)

# Make a prediction on new data
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Change the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
print('Input data as numpy array:', input_data_as_numpy_array)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
print('Input data reshaped:', input_data_reshaped)

# Standardize the input data
std_data = scaler.transform(input_data_reshaped)
print('Standardized input data:', std_data)

# Make the prediction
prediction = classifier.predict(std_data)
print('Prediction:', prediction)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
