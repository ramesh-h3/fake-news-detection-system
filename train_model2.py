import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# --- Load the datasets ---
# Download these files from a public dataset like the Kaggle Fake News Dataset
true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')

# --- Data Preprocessing and Labeling ---
# Add a 'label' column to each dataframe
true_data['label'] = 1  # 1 for real news
fake_data['label'] = 0  # 0 for fake news

# Combine the datasets
news_data = pd.concat([true_data, fake_data]).reset_index(drop=True)

# Shuffle the data
news_data = news_data.sample(frac=1).reset_index(drop=True)

# Drop unused columns
news_data = news_data.drop(['title', 'subject', 'date'], axis=1)

# --- Feature Engineering ---
# Use TF-IDF to convert text into numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(news_data['text'])
y = news_data['label']

# --- Train the model ---
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Evaluation (Optional but recommended) ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# --- Save the model and vectorizer ---
# Save the trained model
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")