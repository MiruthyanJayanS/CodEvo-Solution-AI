import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.datasets import mnist # type: ignore

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to a range of [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten the images into 1D vectors for logistic regression
train_images_flattened = train_images.reshape(len(train_images), -1)
test_images_flattened = test_images.reshape(len(test_images), -1)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images_flattened, train_labels, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
log_reg = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
log_reg.fit(X_train, y_train)

# Predict on validation data
y_val_pred = log_reg.predict(X_val)

# Compute accuracy
accuracy_val = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy_val:.4f}')

# Generate classification report
print(classification_report(y_val, y_val_pred))

# Predict on test data
y_test_pred = log_reg.predict(test_images_flattened)

# Compute test accuracy
accuracy_test = accuracy_score(test_labels, y_test_pred)
print(f'Test Accuracy: {accuracy_test:.4f}')

# Generate classification report
print(classification_report(test_labels, y_test_pred))


# Plotting the results
accuracies = [accuracy_val, accuracy_test]
labels = ['Validation Accuracy', 'Test Accuracy']

plt.bar(labels, accuracies, color=['blue', 'green'])
plt.ylim(0, 1)  # Set y-axis limits to 0-1 to represent accuracy as a percentage
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

import joblib

# Save model weights
joblib.dump(log_reg, 'logistic_regression_mnist_model.pkl')
