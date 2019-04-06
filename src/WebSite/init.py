from flask import Flask, render_template
app = Flask(__name__)


@app.route("/")
def home():
    return "hello"


@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return url_for('static', filename='index.html')


@app.route("/about")
def about():
    return "about"


if __name__ == '__main__':
    app.run(debug=True)
