import json

from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def read_prediction_to_memory():
    global data
    # path to prediction.json
    json_path = "./prediction.json"
    # read json file to dict
    with open(json_path, 'r') as f:
        data = json.load(f)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/1A/24hours', methods=['GET'])
def get_1A_24hours():
    read_prediction_to_memory()
    # return data['1A']
    prediction = data['1A']
    # convert to a dict consists of index and value
    prediction = {i: prediction[i] for i in range(24)}

    return json.dumps(prediction)


@app.route('/1A/3days', methods=['GET'])
def get_1A_3days():
    read_prediction_to_memory()
    # return data['1A']
    prediction = data['1A']
    # convert to a dict consists of index and value
    prediction = {i: prediction[i] for i in range(24 * 3)}

    return json.dumps(prediction)


@app.route('/1A/7days', methods=['GET'])
def get_1A_7days():
    read_prediction_to_memory()
    # return data['1A']
    prediction = data['1A']
    # convert to a dict consists of index and value
    prediction = {i: prediction[i] for i in range(24 * 7)}

    return json.dumps(prediction)


@app.route('/1B/24hours', methods=['GET'])
def get_1B_24hours():
    read_prediction_to_memory()
    prediction = data['1B']
    prediction = {i: prediction[i] for i in range(24)}

    return json.dumps(prediction)


@app.route('/1B/3days', methods=['GET'])
def get_1B_3days():
    read_prediction_to_memory()
    prediction = data['1B']
    prediction = {i: prediction[i] for i in range(24 * 3)}

    return json.dumps(prediction)


@app.route('/1B/7days', methods=['GET'])
def get_1B_7days():
    read_prediction_to_memory()
    prediction = data['1B']
    prediction = {i: prediction[i] for i in range(24 * 7)}

    return json.dumps(prediction)


@app.route('/1C/24hours', methods=['GET'])
def get_1C_24hours():
    read_prediction_to_memory()
    prediction = data['1C']
    prediction = {i: prediction[i] for i in range(24)}

    return json.dumps(prediction)


@app.route('/1C/3days', methods=['GET'])
def get_1C_3days():
    read_prediction_to_memory()
    prediction = data['1C']
    prediction = {i: prediction[i] for i in range(24 * 3)}

    return json.dumps(prediction)


@app.route('/1C/7days', methods=['GET'])
def get_1C_7days():
    read_prediction_to_memory()
    prediction = data['1C']
    prediction = {i: prediction[i] for i in range(24 * 7)}

    return json.dumps(prediction)


@app.route('/1D/24hours', methods=['GET'])
def get_1D_24hours():
    read_prediction_to_memory()
    prediction = data['1D']
    prediction = {i: prediction[i] for i in range(24)}

    return json.dumps(prediction)


@app.route('/1D/3days', methods=['GET'])
def get_1D_3days():
    read_prediction_to_memory()
    prediction = data['1D']
    prediction = {i: prediction[i] for i in range(24 * 3)}

    return json.dumps(prediction)


@app.route('/1D/7days', methods=['GET'])
def get_1D_7days():
    read_prediction_to_memory()
    prediction = data['1D']
    prediction = {i: prediction[i] for i in range(24 * 7)}

    return json.dumps(prediction)


@app.route('/1E/24hours', methods=['GET'])
def get_1E_24hours():
    read_prediction_to_memory()
    prediction = data['1E']
    prediction = {i: prediction[i] for i in range(24)}

    return json.dumps(prediction)


@app.route('/1E/3days', methods=['GET'])
def get_1E_3days():
    read_prediction_to_memory()
    prediction = data['1E']
    prediction = {i: prediction[i] for i in range(24 * 3)}

    return json.dumps(prediction)


@app.route('/1E/7days', methods=['GET'])
def get_1E_7days():
    read_prediction_to_memory()
    prediction = data['1E']
    prediction = {i: prediction[i] for i in range(24 * 7)}

    return json.dumps(prediction)


@app.route('/2A/24hours', methods=['GET'])
def get_2A_24hours():
    read_prediction_to_memory()
    prediction = data['2A']
    prediction = {i: prediction[i] for i in range(24)}

    return json.dumps(prediction)


@app.route('/2A/3days', methods=['GET'])
def get_2A_3days():
    read_prediction_to_memory()
    prediction = data['2A']
    prediction = {i: prediction[i] for i in range(24 * 3)}

    return json.dumps(prediction)


@app.route('/2A/7days', methods=['GET'])
def get_2A_7days():
    read_prediction_to_memory()
    prediction = data['2A']
    prediction = {i: prediction[i] for i in range(24 * 7)}

    return json.dumps(prediction)


@app.route('/2B/24hours', methods=['GET'])
def get_2B_24hours():
    read_prediction_to_memory()
    prediction = data['2B']
    prediction = {i: prediction[i] for i in range(24)}

    return json.dumps(prediction)


@app.route('/2B/3days', methods=['GET'])
def get_2B_3days():
    read_prediction_to_memory()
    prediction = data['2B']
    prediction = {i: prediction[i] for i in range(24 * 3)}

    return json.dumps(prediction)


@app.route('/2B/7days', methods=['GET'])
def get_2B_7days():
    read_prediction_to_memory()
    prediction = data['2B']
    prediction = {i: prediction[i] for i in range(24 * 7)}

    return json.dumps(prediction)


@app.route('/2C/24hours', methods=['GET'])
def get_2C_24hours():
    read_prediction_to_memory()
    prediction = data['2C']
    prediction = {i: prediction[i] for i in range(24)}

    return json.dumps(prediction)


@app.route('/2C/3days', methods=['GET'])
def get_2C_3days():
    read_prediction_to_memory()
    prediction = data['2C']
    prediction = {i: prediction[i] for i in range(24 * 3)}

    return json.dumps(prediction)


@app.route('/2C/7days', methods=['GET'])
def get_2C_7days():
    read_prediction_to_memory()
    prediction = data['2C']
    prediction = {i: prediction[i] for i in range(24 * 7)}

    return json.dumps(prediction)


@app.route('/2D/24hours', methods=['GET'])
def get_2D_24hours():
    read_prediction_to_memory()
    prediction = data['2D']
    prediction = {i: prediction[i] for i in range(24)}

    return json.dumps(prediction)


@app.route('/2D/3days', methods=['GET'])
def get_2D_3days():
    read_prediction_to_memory()
    prediction = data['2D']
    prediction = {i: prediction[i] for i in range(24 * 3)}

    return json.dumps(prediction)


@app.route('/2D/7days', methods=['GET'])
def get_2D_7days():
    read_prediction_to_memory()
    prediction = data['2D']
    prediction = {i: prediction[i] for i in range(24 * 7)}

    return json.dumps(prediction)


@app.route('/2E/24hours', methods=['GET'])
def get_2E_24hours():
    read_prediction_to_memory()
    prediction = data['2E']
    prediction = {i: prediction[i] for i in range(24)}

    return json.dumps(prediction)


@app.route('/2E/3days', methods=['GET'])
def get_2E_3days():
    read_prediction_to_memory()
    prediction = data['2E']
    prediction = {i: prediction[i] for i in range(24 * 3)}

    return json.dumps(prediction)


@app.route('/2E/7days', methods=['GET'])
def get_2E_7days():
    read_prediction_to_memory()
    prediction = data['2E']
    prediction = {i: prediction[i] for i in range(24 * 7)}

    return json.dumps(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
