import ccxt.pro
from asyncio import run
import asyncio
print('CCXT Version:', ccxt.__version__)

def table(values):
    first = values[0]
    keys = list(first.keys()) if isinstance(first, dict) else range(0, len(first))
    widths = [max([len(str(v[k])) for v in values]) for k in keys]
    string = ' | '.join(['{:<' + str(w) + '}' for w in widths])
    return "\n".join([string.format(*[str(v[k]) for k in keys]) for v in values])


async def watch_OHLCV(exchange,
               symbol = "USD/SGD",
               timeframe = '1m',
               limit = 10):

    method = 'watchOHLCV'

    if (method in exchange.has) and exchange.has[method]:
        max_iterations = 100
        for i in range(0, max_iterations):
            try:
                ohlcvs = await exchange.watch_ohlcv(symbol, timeframe, None, limit)
                now = exchange.milliseconds()
                print('\n===============================================================================')
                print('Loop iteration:', i, 'current time:', exchange.iso8601(now), symbol, timeframe)
                print('-------------------------------------------------------------------------------')
                print(table([[exchange.iso8601(o[0])] + o[1:] for o in ohlcvs]))
            except Exception as e:
                print(type(e).__name__, str(e))
                break
        await exchange.close()
    else:
        print(exchange.id, method, 'is not supported or not implemented yet')


async def watch_order_book(exchange,
                           symbol = "USD/SGD",
                           limit = 10):
    if exchange.has['watchOrderBook']:
        while True:
            try:
                orderbook = await exchange.watch_order_book(symbol, limit)
                print(exchange.iso8601(exchange.milliseconds()), symbol, orderbook['asks'][0], orderbook['bids'][0])
            except Exception as e:
                print(e)
                # stop the loop on exception or leave it commented to retry
                # raise e
        await exchange.close()
    else:
        print(exchange.id, 'watchOrderBook not supported')


async def main():

    exchange = ccxt.pro.currencycom()
    symbol = "USD/SGD"
    timeframe = '1m'


    await asyncio.gather(
        watch_OHLCV(exchange, symbol, limit=10),
        watch_order_book(exchange, symbol, limit=10)
    )


run(main())
# %%
