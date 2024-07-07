import aiohttp
import os
import pytz
from datetime import datetime
import numpy as np
import requests
import json

from settings import WEBHOOK_URL
from utils.get_time import get_dt_times_now
from logger.logger import setup_logger

logger = setup_logger(logger_file_name="trading_bot")



def send_msg(message = {'text': 'test'}):

    # Send the message
    response = requests.post(
        WEBHOOK_URL, 
        data=json.dumps(message),
        headers={'Content-Type': 'application/json'}
    )

    # Check the response
    if response.status_code != 200:
        raise ValueError(
            'Request to slack returned an error %s, the response is:\n%s'
            % (response.status_code, response.text)
        )



def send_msg_trade_update(model_state, html_file_path = None,):

    """
    format: BUY, USD1000 of USD/SGD at ref 1.3456
    f"{}
    
    """
    params = {}
    buySignalTitle = '*ENTRY*'
    sellSignalTitle = '*EXIT*'
    signal_alarm = False
    tradable_times = model_state["tradable_times"]
    tradable_times_str = " & ".join([f"{i[0]} to {i[-1]}" for i in tradable_times])
    warning_symbol = ":warning:"

    for pos in ["L","S"]:
        signal_alarm = False
        signalTime = model_state[pos]["time"]
        updateTime = model_state[pos]["time"]

        if model_state[pos]["entry_now"]:
            if pos == "L":
                action_prices = f'expected entry price ------ {model_state[pos]["entry_price_expected"]}\n' \
                                f'expected entry price max -- {model_state[pos]["entry_price_max"]}\n' \
                                f'model trade id ------------ {model_state[pos]["model_id"]} \n' \
                                f'* actual entry price ------ {model_state[pos]["entry_price"]} \n' \
                                f'* actual trade id --------- {model_state[pos]["order_id"]}'
                pos_name = "Long"
                pos_symbol = ':large_green_square:'
            elif pos == "S":
                action_prices = f'expected entry price ----- {model_state[pos]["entry_price_expected"]}\n' \
                                f'expected entry price min -- {model_state[pos]["entry_price_min"]}\n' \
                                f'model trade id ----------- {model_state[pos]["model_id"]} \n' \
                                f'* actual entry price ------ {model_state[pos]["entry_price"]} \n' \
                                f'* actual trade id --------- {model_state[pos]["order_id"]}'
                pos_name = "Short"
                pos_symbol = ':large_red_square:'

            signal_alarm = True
            logger.info(f'[{updateTime}] - LONG entry {pos} Signal Detected for - {model_state[pos]["pair"]}')
            params = {
                'type': 'mrkdwn',
                'text': f'{warning_symbol} {pos_symbol} *{pos_name}* *Trade* Update --> {buySignalTitle} {warning_symbol}',
                'attachments': [
                    {
                        'type': 'mrkdwn',
                        'color': '#48C9B0',
                        'fields': [
                            {
                                'title': f' {buySignalTitle} {pos_name} (signal: {model_state["running_signal"]})',
                                'value': f'{action_prices}',
                                'short': True
                            },
                            {
                                'title': '*TRADE TIME*',
                                'value': f'{signalTime}',
                                'short': True
                            },
                        ],
                        'footer': f'*Signal Time* ' + signalTime + '\n*Tradable Times* ' + tradable_times_str,
                    }
                ],
            }

        elif model_state[pos]["exit_now"]:
            if pos == "L":
                action_prices = f'expected exit price ------ {model_state[pos]["exit_price_expected"]}\n' \
                                f'expected exit price min -- {model_state[pos]["exit_price_min"]}\n' \
                                f'model trade id ----------- {model_state[pos]["model_id"]} \n' \
                                f'* actual exit price ------ {model_state[pos]["exit_price"]} \n' \
                                f'* actual trade id --------- {model_state[pos]["order_id"]}'
                
                pos_name = "Long"
                pos_symbol = ':large_green_square:'
            elif pos == "S":
                action_prices = f'expected exit price ------ {model_state[pos]["exit_price_expected"]}\n' \
                                f'expected exit price max -- {model_state[pos]["exit_price_max"]}\n' \
                                f'model trade id ----------- {model_state[pos]["model_id"]} \n' \
                                f'* actual exit price ------ {model_state[pos]["exit_price"]} \n' \
                                f'* actual trade id --------- {model_state[pos]["order_id"]}'
                pos_name = "Short"
                pos_symbol = ':large_red_square:'

            signal_alarm = True
            logger.info(f'[{updateTime}] - SHORT exit {pos} Signal Detected for - {model_state[pos]["pair"]}')
            params = {
                'type': 'mrkdwn',
                'text': f'{warning_symbol} {pos_symbol} *{pos_name}* *Trade* Update --> {sellSignalTitle} {warning_symbol}',
                'attachments': [
                    {
                        'type': 'mrkdwn',
                        'color': '#E74C3C',
                        'fields': [
                            {
                                'title': f'{sellSignalTitle} {pos_name} (signal: {model_state["running_signal"]})',
                                'value': f'{action_prices}',
                                'short': True
                            },
                            {
                                'title': '*TRADE TIME*',
                                'value': f'{signalTime}',
                                'short': True
                            },
                        ],
                        'footer': f'*Signal Time* ' + signalTime + '\n*Tradable Times* ' + tradable_times_str,
                    }
                ],
            }

        if signal_alarm: 
            requests.post(WEBHOOK_URL, json=params)
            logger.info(f'Slack notification sent! {updateTime}')
            signal_alarm = False
        else: 
            logger.info(f'[{updateTime}] - No Signal for {model_state[pos]["pair"]} | Last window [{signalTime}]')
    # Upload HTML file to Slack
    publish_plotly_html(model_state, html_file_path)



def send_msg_position_update(model_state, html_file_path=None):
    # updateTime, updateTimeSGT = get_dt_times_now()
    params = {}
    for pos in ["L","S"]:
        updateTime = model_state[pos]["time"]
        tradable_times = model_state[pos]["tradable_times"]
        if tradable_times is None:
            tradable_times_str = "None"
        else:
            tradable_times_str = " & ".join([f"{i[0]} to {i[-1]}" for i in tradable_times])

        if pos == "L":
            pos_symbol = ':large_green_square:'
            pos_name = "Long"
            action_prices = f'expected entry price ------ {model_state[pos]["entry_price_expected"]}\n' \
                            f'expected entry price max -- {model_state[pos]["entry_price_max"]}\n' \
                            f'model trade id ------------ {model_state[pos]["model_id"]} \n' \
                            f'* actual entry price ------ {model_state[pos]["entry_price"]} \n' \
                            f'* actual trade id --------- {model_state[pos]["order_id"]}'
        elif pos == "S":
            pos_name = "Short"
            pos_symbol = ':large_red_square:'
            action_prices = f'expected entry price ------ {model_state[pos]["entry_price_expected"]}\n' \
                            f'expected entry price min -- {model_state[pos]["entry_price_min"]}\n' \
                            f'model trade id ------------ {model_state[pos]["model_id"]}\n' \
                            f'* actual entry price ------ {model_state[pos]["entry_price"]}\n' \
                            f'* actual trade id --------- {model_state[pos]["order_id"]}'

        else:
            pos_name = "Flat"
            pos_symbol = ':large_black_square:'
        if model_state[pos]["in_position"] and model_state[pos][f"positions"] != 0:
            params = {
                'type': 'mrkdwn',
                'text': f'*{pos_symbol}* *{pos_name}* *Position* Update',
                'attachments': [
                    {
                        'type': 'mrkdwn',
                        'color': '#E74C3C',
                        'fields': [
                            {
                                'title': f'*{pos_name} Position*',
                                'value': f'{action_prices}',
                                'short': True
                            },
                            {
                                'title': '*ENTRY TIME*',
                                'value': f'{model_state[pos][f"entry_time"]}',
                                'short': True
                            },

                        ],
                        'footer': f'*Update Time* ' + updateTime + '\n*Tradable Times* ' + tradable_times_str +' UTC'
                    }
                ],
            }
        else:
            params = {
                'type': 'mrkdwn',
                'text': f'*{pos_symbol}* *{pos_name}* *Position* Update',
                'attachments': [
                    {
                        'type': 'mrkdwn',
                        'color': '#E74C3C',
                        'fields': [
                            {
                                'title': f'NO positions open',
                                'value': '',
                                'short': True
                            },
                        ],
                        'footer': f'*Update Time* ' + updateTime + '\n*Tradable Times* ' + tradable_times_str +' UTC'
                    }
                ],
            }

        requests.post(WEBHOOK_URL, json=params)
        print(f'Slack notification sent! {updateTime}')

            # Upload HTML file to Slack
    publish_plotly_html(model_state, html_file_path)



def publish_plotly_html(model_state, html_file_path):
    # Upload HTML file to Slack
    if html_file_path:

        file_upload = {
            "file": (html_file_path, open(html_file_path, 'rb'), 'text/html'),
        }
        response = requests.post(WEBHOOK_URL, files=file_upload)
        print(response.json())
        if not response.json().get('ok'):
            raise Exception(f"Failed to upload HTML file: {response.json().get('error')}")
        




















# def format_message(model_state, pos, buySignalTitle, sellSignalTitle):
#     # print(f"MODEL STATE: \n{model_state}\n")
#     params = {}
#     signalTime = model_state[pos]["time"]
#     if model_state[pos]["entry_now"]:
#         filled_prices = f'filled entry price: {model_state[pos]["entry_price_filled"]}\nfilled entry price: {model_state[pos]["entry_price_filled"]}'
#         if pos == "L":
#             action_prices = f'target entry price: {model_state[pos]["entry_price"]}\nmax entry price: {model_state[pos]["entry_price_max"]}'
#             pos_name = "Long"
#         elif pos == "S":
#             action_prices = f'target entry price: {model_state[pos]["entry_price"]}\nmin entry price: {model_state[pos]["entry_price_min"]}'
#             pos_name = "Short"
#         signal_alarm = True
#         print(f'[{updateTime}] - 🔼 entry {pos} Signal Detected for - {model_state[pos]["pair"]}')
#         params = {
#             'type': 'mrkdwn',
#             'text': f':vertical_traffic_light: {model_state[pos]["model"]} {model_state[pos]["pair"]}',
#             'attachments': [
#                 {
#                     'type': 'mrkdwn',
#                     'color': '#48C9B0',
#                     'fields': [
#                         {
#                             'title': '*Signal*',
#                             'value': f'{model_state["running_signal"]} --> {buySignalTitle} {pos_name}\n{filled_prices}',
#                             'short': True
#                         },
#                         {
#                             'title': '*Asset*',
#                             'value': f'{action_prices}',
#                             'short': True
#                         },
#                         {
#                             'title': '*Take Profits*',
#                             'value': f'TP1 @ {model_state[pos]["TP1"]}' + f'\nTP2 @ {model_state[pos]["TP2"]}',
#                             'short': True
#                         },
#                         {
#                             'title': '*Stop Loss Levels*',
#                             'value': f'SL1 @ {model_state[pos]["SL1"]}' + f'\nSL2 @ {model_state[pos]["SL2"]}',
#                             'short': True
#                         },
#                     ],
#                     'footer': '*Signal Time* ' + signalTime + '\n*Tradable Times* ' + " & ".join([f"{i[0]} to {i[-1]}" for i in model_state["tradable_times"]]),
#                     'footer_icon': 'https://s3-ap-southeast-1.amazonaws.com/call-levels.com/email+signatures/logo_email.png'
#                 }
#             ],
#         }
#     return params