import pytz
from datetime import datetime
import requests
import os

def checkWeekday():
    # Add your implementation here
    pass

def pushSignal(res, val):
    sg_tz = pytz.timezone('Asia/Singapore')
    now = datetime.now(sg_tz)
    interval = int(now.strftime('%M'))
    outputTime = now.strftime('%a %d %b %Y %I:%M:%S %p')
    currentTime = outputTime

    pair = val
    params = {}
    signal = {}

    signal = {
        'pair': pair,
        'time': res[pair]['time'],
        'price': res[pair]['price'],
        'signal': res[pair]['signal'],
        'position': int(res[pair]['position']),
        'upper_limit': res[pair]['upper_limit'],
        'lower_limit': res[pair]['lower_limit'],
    }

    london_tz = pytz.timezone('Europe/London')
    signalTime = datetime.strptime(signal['time'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=london_tz)
    convertedSignalTime = signalTime.astimezone(sg_tz).strftime('%a %d %b %Y %I:%M:%S %p')

    buySignalTitle = ':arrow_up: *BUY*'
    sellSignalTitle = ':arrow_down: *SELL*'

    if signal['position'] > 0:
        print(f'[{outputTime}] - 🔼 Buy Signal Detected for - {signal["pair"]}')
        params = {
            'type': 'mrkdwn',
            'text': ':vertical_traffic_light: Trading Signal :moneybag:', 
            'attachments': [
            {
                'type': 'mrkdwn',
                'color': '#48C9B0',
                'fields': [
                {
                    'title': '*Signal*',
                    'value': buySignalTitle,
                    'short': True
                },
                {
                    'title': '*Asset*',
                    'value': signal['pair'] + ' @ ' + str(signal['price']),
                    'short': True
                },
                {
                    'title': '*Take Profit Level*',
                    'value': 'TP @ ' + str(signal['upper_limit']),
                    'short': True
                },
                {
                    'title': '*Stop Loss Level*',
                    'value': 'SL @ ' + str(signal['lower_limit']),
                    'short': True
                },
                ],
                'footer': '*Signal Time* ' + convertedSignalTime,
                'footer_icon': 'https://s3-ap-southeast-1.amazonaws.com/call-levels.com/email+signatures/logo_email.png'
            }],
        }
    elif signal['position'] < 0:
        print(f'[{outputTime}] - 🔽 Sell Signal Detected for - {signal["pair"]}')
        params = {
            'type': 'mrkdwn',
            'text': ':vertical_traffic_light: Trading Signal :moneybag:', 
            'attachments': [
            {
                'type': 'mrkdwn',
                'color': '#E74C3C',
                'fields': [
                {
                    'title': '*Signal*',
                    'value': sellSignalTitle,
                    'short': True
                },
                {
                    'title': '*Asset*',
                    'value': signal['pair'] + ' @' + str(signal['price']),
                    'short': True
                },
                {
                    'title': '*Take Profit Level*',
                    'value': "TP @ " + str(signal['lower_limit']),
                    "short": True
                },
                {
                    "title": "*Stop Loss Level*",
                    "value": "SL @ " + str(signal['upper_limit']),
                    "short": True
                },
                ],
                "footer": "*Signal Time* " + convertedSignalTime,
                "footer_icon": "https://s3-ap-southeast-1.amazonaws.com/call-levels.com/email+signatures/logo_email.png"
            }],
        }
    elif signal['position'] == 0:
        print(f'[{currentTime}] - ❌ No Signal Detected for - {signal["pair"]} - Last Window[{convertedSignalTime}]')

    try:
        if checkWeekday():
            if signal['position'] != 0 and interval == 5:
                response = requests.post(os.getenv('SLACK_SIGNAL_WEBHOOK_PUBLIC'), json=params)
                if response.status_code == 200:
                    print(f'Slack notification sent! {outputTime}')
                    requests.post(os.getenv('SLACK_SIGNAL_WEBHOOK_TEST'), json=params)
            elif signal['position'] == 0 and interval == 0:
                print(f'[{outputTime}] - No Signal for {signal["pair"]} | Last window [{convertedSignalTime}]')
    except Exception as error:
        print(error)


#%% 
if __name__ == "__main__":
    # https://hooks.slack.com/services/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
    pushSignal