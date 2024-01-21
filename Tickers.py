import asyncio
import datetime
import json

import websockets


async def azz_ws():
    url = 'wss://ws.okx.com:8443/ws/v5/public'
    async with websockets.connect(url, ping_interval=20, ping_timeout=60) as ws:
        print('Connected ' + datetime.datetime.now().isoformat())

        subs = dict(
            op='subscribe',
            args=[
                dict(channel='mark-price', instId='BTC-USDT')
            ]
        )
        await ws.send(json.dumps(subs))

        async for msg in ws:
            msg = json.loads(msg)
            ev = msg.get('event')
            data = msg.get('data')
            if ev:
                print(f"************* event {ev} = {msg.get('arg')} *************")
            elif data and len(data) > 0:
                symbol = data[0].get('instId')
                price = float(data[0].get('markPx', '0.0'))
                print(f"{datetime.datetime.now().isoformat()} {symbol} = {price}")
                return

print('Disconnected ' + datetime.datetime.now().isoformat())

if __name__ == '__main__':
    print(f"AzzraelCode YouTube Subs")
    asyncio.run(azz_ws())