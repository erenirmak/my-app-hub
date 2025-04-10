from flask import Flask, render_template
import json

app = Flask(__name__)

@app.route('/')
def index():

    with open("apps.json") as f:
        applications = json.load(f)

    with open("collapsible.json", "r") as f:
        collapsibles = json.load(f)

    return render_template('index.html', applications=applications, collapsibles=collapsibles)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)