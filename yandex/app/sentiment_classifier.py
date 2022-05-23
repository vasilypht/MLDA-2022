import joblib
import re

from pymystem3 import Mystem
from pyaspeller import YandexSpeller


class SentimentClassifier(object):
    def __init__(self):
        self.messages = {
                            0: "Отрицательный отзыв :(",
                            1: "Положительный отзыв :)" 
                        }

        self.vectorizer = joblib.load("tfidf_vectorizer.pkl")
        self.clf = joblib.load("linsvc_clf.pkl")

        self.regex_spec_char = re.compile(r"\W")
        self.regex_multi_spaces = re.compile(r"\s+")
        self.regex_digits = re.compile(r"\d+")

        self.lem = Mystem()
        self.speller = YandexSpeller()
        
    def predict(self, sentence: str):
        sentence = self.speller.spelled(sentence)

        sentence = sentence.lower()
        sentence = self.regex_spec_char.sub(" ", sentence)
        sentence = self.regex_digits.sub(" ", sentence)
        sentence = self.regex_multi_spaces.sub(" ", sentence)

        sentence = "".join(self.lem.lemmatize(sentence))
        predict = self.clf.predict(self.vectorizer.transform([sentence]))
        return self.messages.get(predict[0])
