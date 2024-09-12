import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import joblib

# Example dataset
data = {
    'text': ["Win a free iPhone now!", "Your order is confirmed", "Congratulations, you've won!", 
             "Meeting at 5 PM", "Free vacation offer!", "Project deadline is tomorrow"],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split data into features (text) and labels
X = df['text']
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Transform the text data into TF-IDF vectors
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = nb_model.predict(X_test_tfidf)

# Compute accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

# Generate classification report
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(nb_model, 'naive_bayes_text_classifier.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
