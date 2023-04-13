# SDP-workplace

This is the repo. for my SDP: electricity forecast (ELF) device.

Make sure anaconda is installed on your machine.

Proxy is required on PORT 7890, otherwise you can set the _**port**_ parameter of weather_crawler.

Create the conda virtual environment.

` conda env create -n ece445_elf -f SDP_env.yaml `

To generate the prediction of hourly electricity usage in next week, run

`python main.py`

The file _prediction.json_ contains the result for each building's next week hourly electricity usage.
