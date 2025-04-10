from flask import Flask, render_template
import json

app = Flask(__name__)

@app.route('/')
def index():
    mlops = "https://github.com/graviraja/MLOps-Basics"

    applications = [
        {'name': 'İstanbul Sanayi Odası - Agent Test', 'url': 'http://10.0.0.1:8501'},
        {'name': 'Eksim Holding - Agent Test', 'url': 'http://10.0.0.1:8502'},
        {'name': 'Agent App Tracing', 'url': 'http://10.0.0.1:6006'},
        {'name': 'Airflow', 'url': 'http://10.0.0.1:8080/home'},
        {'name': 'Jupyter Server', 'url': 'http://10.0.0.1:8888/tree'},
    ]
    
    workplace = {
        'name': 'BookStack Workplace', 'url': 'http://10.0.0.1:6875'
    }
    with open("collapsible.json", "r") as f:
        collapsibles = json.load(f)

    return render_template('index.html', workplace=workplace, applications=applications, collapsibles=collapsibles)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)