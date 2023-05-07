from utils import electricity_complete_api
import pandas as pd


def main():
    api = electricity_complete_api()
    api.complete_electricity()


if __name__ == '__main__':
    main()
