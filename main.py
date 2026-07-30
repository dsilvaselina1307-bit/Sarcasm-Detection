import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("dataset/sarcasm_data.csv")

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["text"])

# Labels
y = data["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Test
sentence = input("Enter a sentence: ")
test = vectorizer.transform([sentence])

prediction = model.predict(test)

if prediction[0] == 1:
    print("Prediction: Sarcastic")
else:
    print("Prediction: Not Sarcastic")