# SDP-workplace

This is the repo. for my SDP: electricity forecast (ELF) device.

Make sure anaconda is installed on your machine.

Proxy is required on PORT 7890, otherwise you can set the _**port**_ parameter of weather_crawler.

Create the conda virtual environment.

Package request:

`   pytorch-forecasting=0.10.2 pytorch-lightning=1.7.0 pytorch=1.13.1`

Activate the virtual environment

` conda activate DeepAR_env`

Go to the prediction

` cd prediction_module/code/`

To generate the prediction of hourly electricity usage in next week, run

`python main.py`

The file _prediction.json_ contains the result for each building's next week hourly electricity usage.
