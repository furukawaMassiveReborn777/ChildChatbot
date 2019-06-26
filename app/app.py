# coding:utf-8
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import nltk
from answer import answer
nltk.download('punkt')# only first time
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__, static_folder='static')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/post', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        question = request.form['question']
        return render_template('index.html', question=question, answer=answer(question))
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)