from flask import Flask, render_template, jsonify
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model-metrics')
def metrics():
    report_path = os.path.join(REPORT_DIR, "classification_report.txt")
    try:
        with open(report_path, 'r') as f:
            report = f.read()
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
