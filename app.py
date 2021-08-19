# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
#filename = 'voting_clf.pkl'
NB = pickle.load(open('Review.pkl', 'rb'))
cv = pickle.load(open('countvector.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
   
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST': 
      Review = request.form['Review']
      data = [Review]
      vect = cv.transform(data).toarray()
      my_prediction = NB.predict(vect)
      return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)