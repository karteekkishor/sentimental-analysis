from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

data = pd.read_csv("data6.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Sentence'])
y = data['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifier = LinearSVC()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

app = Flask(__name__)

db_config = { 
    'host': "127.0.0.1",
    'user': 'root',
    'password': 'root',
    'database': 'svecdb',
}
@app.route('/')
def home():
    return render_template('ind.html')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_input= request.form['q1'] 
        user_input_vectorized = vectorizer.transform([user_input])
        user_sentiment = classifier.predict(user_input_vectorized)
        print(user_sentiment[0])
        return render_template('ind.html',sentiment=user_sentiment[0],d=user_input)
    return render_template('ind.html')

if __name__ == '__main__':
    app.run(debug=True)  



















