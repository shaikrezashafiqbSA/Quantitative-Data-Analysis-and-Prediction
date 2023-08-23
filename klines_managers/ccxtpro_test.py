import ccxt.pro
from asyncio import run

print('CCXT Pro version', ccxt.pro.__version__)


def table(values):
    first = values[0]
    keys = list(first.keys()) if isinstance(first, dict) else range(0, len(first))
    widths = [max([len(str(v[k])) for v in values]) for k in keys]
    string = ' | '.join(['{:<' + str(w) + '}' for w in widths])
    return "\n".join([string.format(*[str(v[k]) for k in keys]) for v in values])


# async def _watchOHLCV(symbol='BTC/USDT',
#                       timeframe="1m",
#                       limit=10,
#                       exchange_name="okx"):
#     exchange = getattr(ccxt.pro, exchange_name)
#     method = 'watchOHLCV'
#     if (method in exchange.has) and exchange.has[method]:
#         max_iterations = 100000  # how many times to repeat the loop before exiting
#         for i in range(0, max_iterations):
#             try:
#                 ohlcvs = await exchange.watch_ohlcv(symbol, timeframe, None, limit)
#                 now = exchange.milliseconds()
#                 print('\n===============================================================================')
#                 print('Loop iteration:', i, 'current time:', exchange.iso8601(now), symbol, timeframe)
#                 print('-------------------------------------------------------------------------------')
#                 print(table([[exchange.iso8601(o[0])] + o[1:] for o in ohlcvs]))
#             except Exception as e:
#                 print(type(e).__name__, str(e))
#                 break
#         await exchange.close()
#     else:
#         print(exchange.id, method, 'is not supported or not implemented yet')

import ccxt.pro
import asyncio
from asyncio import Queue

async def main():
    exchange = ccxt.pro.binance({
        'enableRateLimit': True,
        'asyncio_loop': asyncio.get_event_loop(),
    })

    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    timeframes = ['1m', '5m', '15m']

    queue = Queue()

    async def handle_ohlcv(symbol, timeframe, ohlcv):
        await queue.put((symbol, timeframe, ohlcv))

    for symbol in symbols:
        for timeframe in timeframes:
            await exchange.watch_ohlcv(symbol, timeframe, handle_ohlcv)

    while True:
        symbol, timeframe, ohlcv = await queue.get()
        print(f'{symbol} {timeframe} {ohlcv}')

if __name__ == '__main__':
    asyncio.run(main())







# if __name__ == "__main__":
#     run(_watchOHLCV())
    
    
    # #%%
    # import ccxt.pro 
    # exchange_name = "okx"
    # exchange = getattr(ccxt.pro, exchange_name)