import json
import datetime
import os
import sys

from flask import Flask
from flask_cors import CORS, cross_origin

from pred_api import prediction_api





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
    # create a log file to record the print
    return 'Hello World!'


# a Get request with two parameters: the first is the name of the building, the second is time interval,
# the time interval may be 24hours, 3days or 7days. The prediction is returned in the format of a json string.
@app.route('/<building>/<time_interval>', methods=['GET'])
def get_prediction(building, time_interval):
    read_prediction_to_memory()
    prediction = data[building]
    if time_interval == '24hours':
        prediction = {i: prediction[i] for i in range(24)}
    elif time_interval == '3days':
        prediction = {i: prediction[i] for i in range(24 * 3)}
    elif time_interval == '7days':
        prediction = {i: prediction[i] for i in range(24 * 7)}
    else:
        return 'Invalid time interval'

    return json.dumps(prediction)


# a Get request with two parameters: the first is the name of the building, the second is the start date of the
# prediction, the start date is in the format: "%Y-%m-%d" (e.g. 2019-01-01). The prediction is returned in the format
# of a json string.
@app.route('/demo/<building>/<time_interval>/<start_date>', methods=['GET'])
def get_prediction_demo(building, time_interval, start_date):
    # reformat the start_date to the format: "%Y%m%d" (e.g. 20190101)
    start_date = start_date.replace('-', '')

    model_path = "./my_model/hidden=28-rnn_layer=2-context_day=30-min_lr=0.0001.ckpt"
    pred_date_start = datetime.datetime.strptime(start_date, "%Y%m%d")
    num_day_context = 30
    weather_start_date = pred_date_start - datetime.timedelta(days=num_day_context + 1)

    weather_start_date = weather_start_date.strftime("%Y%m%d")
    pred_date_start = pred_date_start.strftime("%Y%m%d")
    # check if files named "../data/output/origin-pred_date={pred_date_start}-weather_date={weather_start_date}.json"
    # and "../data/output/prediction-pred_date={pred_date_start}-weather_date={weather_start_date}.json" both exist
    # if not, run the prediction api
    if not os.path.isfile(
            "../data/output/origin-pred_date=" + pred_date_start + ".json") \
            or not os.path.isfile(
        "../data/output/prediction-pred_date=" + pred_date_start + ".json"):
        predictor = prediction_api()
        predictor.custom_prediction(model_path, pred_date_start, weather_start_date, num_day_context)

    # read in a json file named "./data/output/prediction-pred_date={pred_date_start}-weather_date={weather_start_date}.json"
    json_path = "../data/output/prediction-pred_date=" + pred_date_start + ".json"
    with open(json_path, 'r') as f:
        prediction = json.load(f)
    # read in a json file named "./data/output/origin-pred_date={pred_date_start}-weather_date={weather_start_date}.json"
    json_path = "../data/output/origin-pred_date=" + pred_date_start + ".json"
    with open(json_path, 'r') as f:
        origin = json.load(f)

    # return the prediction and origin for the specified building and time interval in the format of a json string
    if time_interval == '24hours':
        prediction = {i: prediction[building][i] for i in range(24)}
        origin = {i: origin[building][i] for i in range(24)}
    elif time_interval == '3days':
        prediction = {i: prediction[building][i] for i in range(24 * 3)}
        origin = {i: origin[building][i] for i in range(24 * 3)}
    elif time_interval == '7days':
        prediction = {i: prediction[building][i] for i in range(24 * 7)}
        origin = {i: origin[building][i] for i in range(24 * 7)}
    else:
        return 'Invalid time interval'

    return json.dumps({'prediction': prediction, 'real': origin})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
