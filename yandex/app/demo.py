from codecs import open
import time

from flask import (
    Flask,
    render_template,
    request
)

from sentiment_classifier import SentimentClassifier

app = Flask(__name__)  #имя модуля приложения

print ("Подготовка классификатора")
start_time = time.time()
classifier = SentimentClassifier()
print ("Классификатор готов к работе")
print (time.time() - start_time, "seconds")

@app.route("/", methods=["POST", "GET"]) 
def index_page(text = "", prediction_message = ""):
    if request.method == "POST": 
        text = request.form["text"]
        prediction_message = classifier.predict(text)
        print(prediction_message)

    return render_template('index.html', text=text, prediction_message=prediction_message)


if __name__ == '__main__':
    app.run(debug=False, port=8080)
