import multiprocessing
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({"message": "Hello, RR Chatbot!"})

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Add freeze_support for Windows multiprocessing
    multiprocessing.freeze_support()
    
    app.run(host='0.0.0.0', port=5000, debug=True) 