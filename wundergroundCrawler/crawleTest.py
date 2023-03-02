import requests

# 配置代理
proxies = {'http': 'http://127.0.0.1:1001', 'https': 'http://127.0.0.1:1001'}
response = requests.get('https://www.google.com/', proxies=proxies)
print(response.status_code)
