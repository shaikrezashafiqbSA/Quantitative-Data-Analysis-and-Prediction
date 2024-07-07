import json
import websocket

def on_message(ws, message):
  data = json.loads(message)

  if data['type'] == 'PRICE':
    print('USDSGD price:', data['data']['price'])

def on_error(ws, error):
  print('Websocket error:', error)

def on_close(ws):
  print('Websocket closed')

def main():
  ws = websocket.WebSocketApp('wss://api-fxtrade.oanda.com/v20/prices/stream', on_message=on_message, on_error=on_error, on_close=on_close)

  # Subscribe to USDSGD price stream
  ws.send(json.dumps({
    "instruments": ["USDSGD"],
    "type": "subscribe"
  }))

  ws.run_forever()

if __name__ == '__main__':
  import pyautogui
  import time

  # Move the mouse cursor slightly every 10 seconds
  while True:
      pyautogui.moveRel(1, 1)
      time.sleep(10)
