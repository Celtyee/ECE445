from utils import forecast_api
import datetime

num_day_pred = 7
pred_date_start = datetime.datetime.now().date()
pred_date_end = pred_date_start + datetime.timedelta(days=num_day_pred - 1)

forecast_crawler = forecast_api()
forecast_crawler.crawl_forecast(pred_date_start, pred_date_end)
