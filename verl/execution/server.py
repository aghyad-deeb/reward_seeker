from flask import Flask, request

app = Flask(__name__)

@app.route('/message', methods=['POST'])
def receive_message():
    data = request.json  # Get JSON data from sender
    try:
        exec(data["code"])
        return {"status": True}, 200  # Respond to sender
    except:
        return {"status": False}, 200  # Respond to sender

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5555) 
