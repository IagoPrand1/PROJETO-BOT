import asyncio
import websockets
import json
import requests
import hmac
import base64
import zlib
import datetime
import time

import asyncio
import base64
import hmac
import json
import logging
import os
import datetime
import websockets
import pickle
import okx.MarketData as MarketData
import okx.Trade as Trade

marketDataAPI =  MarketData.MarketAPI(flag='1')

def get_timestamp():
    now = datetime.datetime.now()
    t = now.isoformat("T", "milliseconds")
    return t + "Z"


def get_server_time():
    url = "https://www.okx.com/api/v5/private/time"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['data'][0]['ts']
    else:
        return ""

def get_local_timestamp():
    return int(time.time())


def login_params(timestamp, api_key, passphrase, secret_key):
    message = timestamp + 'GET' + '/users/self/verify'

    mac = hmac.new(bytes(secret_key, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
    d = mac.digest()
    sign = base64.b64encode(d)

    login_param = {"op": "login", "args": [{"apiKey": api_key,
                                            "passphrase": passphrase,
                                            "timestamp": timestamp,
                                            "sign": sign.decode("utf-8")}]}
    login_str = json.dumps(login_param)
    return login_str


def partial(res):
    data_obj = res['data'][0]
    bids = data_obj['bids']
    asks = data_obj['asks']
    instrument_id = res['arg']['instId']
    # print('全量数据bids为：' + str(bids))
    # print('档数为：' + str(len(bids)))
    # print('全量数据asks为：' + str(asks))
    # print('档数为：' + str(len(asks)))
    return bids, asks, instrument_id


def update_bids(res, bids_p):
    # 获取增量bids数据
    bids_u = res['data'][0]['bids']
    # print('增量数据bids为：' + str(bids_u))
    # print('档数为：' + str(len(bids_u)))
    # bids合并
    for i in bids_u:
        bid_price = i[0]
        for j in bids_p:
            if bid_price == j[0]:
                if i[1] == '0':
                    bids_p.remove(j)
                    break
                else:
                    del j[1]
                    j.insert(1, i[1])
                    break
        else:
            if i[1] != "0":
                bids_p.append(i)
    else:
        bids_p.sort(key=lambda price: sort_num(price[0]), reverse=True)
        # print('合并后的bids为：' + str(bids_p) + '，档数为：' + str(len(bids_p)))
    return bids_p


def update_asks(res, asks_p):
    # 获取增量asks数据
    asks_u = res['data'][0]['asks']
    # print('增量数据asks为：' + str(asks_u))
    # print('档数为：' + str(len(asks_u)))
    # asks合并
    for i in asks_u:
        ask_price = i[0]
        for j in asks_p:
            if ask_price == j[0]:
                if i[1] == '0':
                    asks_p.remove(j)
                    break
                else:
                    del j[1]
                    j.insert(1, i[1])
                    break
        else:
            if i[1] != "0":
                asks_p.append(i)
    else:
        asks_p.sort(key=lambda price: sort_num(price[0]))
        # print('合并后的asks为：' + str(asks_p) + '，档数为：' + str(len(asks_p)))
    return asks_p


def sort_num(n):
    if n.isdigit():
        return int(n)
    else:
        return float(n)


def check(bids, asks):
    # 获取bid档str
    bids_l = []
    bid_l = []
    count_bid = 1
    while count_bid <= 25:
        if count_bid > len(bids):
            break
        bids_l.append(bids[count_bid-1])
        count_bid += 1
    for j in bids_l:
        str_bid = ':'.join(j[0 : 2])
        bid_l.append(str_bid)
    # 获取ask档str
    asks_l = []
    ask_l = []
    count_ask = 1
    while count_ask <= 25:
        if count_ask > len(asks):
            break
        asks_l.append(asks[count_ask-1])
        count_ask += 1
    for k in asks_l:
        str_ask = ':'.join(k[0 : 2])
        ask_l.append(str_ask)
    # 拼接str
    num = ''
    if len(bid_l) == len(ask_l):
        for m in range(len(bid_l)):
            num += bid_l[m] + ':' + ask_l[m] + ':'
    elif len(bid_l) > len(ask_l):
        # bid档比ask档多
        for n in range(len(ask_l)):
            num += bid_l[n] + ':' + ask_l[n] + ':'
        for l in range(len(ask_l), len(bid_l)):
            num += bid_l[l] + ':'
    elif len(bid_l) < len(ask_l):
        # ask档比bid档多
        for n in range(len(bid_l)):
            num += bid_l[n] + ':' + ask_l[n] + ':'
        for l in range(len(bid_l), len(ask_l)):
            num += ask_l[l] + ':'

    new_num = num[:-1]
    int_checksum = zlib.crc32(new_num.encode())
    fina = change(int_checksum)
    return fina


def change(num_old):
    num = pow(2, 31) - 1
    if num_old > num:
        out = num_old - num * 2 - 2
    else:
        out = num_old
    return out


# subscribe channels un_need login
async def subscribe_without_login(url, channels):
    l = []
    while True:
        try:
            async with websockets.connect(url) as ws:
                sub_param = {"op": "subscribe", "args": channels}
                sub_str = json.dumps(sub_param)
                await ws.send(sub_str)
                print(f"send: {sub_str}")

                while True:
                    try:
                        res = await asyncio.wait_for(ws.recv(), timeout=25)
                    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
                        try:
                            await ws.send('ping')
                            res = await ws.recv()
                            print(res)
                            continue
                        except Exception as e:
                            print("连接关闭，正在重连……")
                            break

                    print(get_timestamp() + res)
                    res = eval(res)
                    if 'event' in res:
                        continue
                    for i in res['arg']:
                        if 'books' in res['arg'][i] and 'books5' not in res['arg'][i]:
                            # 订阅频道是深度频道
                            if res['action'] == 'snapshot':
                                for m in l:
                                    if res['arg']['instId'] == m['instrument_id']:
                                        l.remove(m)
                                # 获取首次全量深度数据
                                bids_p, asks_p, instrument_id = partial(res)
                                d = {}
                                d['instrument_id'] = instrument_id
                                d['bids_p'] = bids_p
                                d['asks_p'] = asks_p
                                l.append(d)

                                # 校验checksum
                                checksum = res['data'][0]['checksum']
                                # print('推送数据的checksum为：' + str(checksum))
                                check_num = check(bids_p, asks_p)
                                # print('校验后的checksum为：' + str(check_num))
                                if check_num == checksum:
                                    print("校验结果为：True")
                                else:
                                    print("校验结果为：False，正在重新订阅……")

                                    # 取消订阅
                                    await unsubscribe_without_login(url, channels)
                                    # 发送订阅
                                    async with websockets.connect(url) as ws:
                                        sub_param = {"op": "subscribe", "args": channels}
                                        sub_str = json.dumps(sub_param)
                                        await ws.send(sub_str)
                                        print(f"send: {sub_str}")

                            elif res['action'] == 'update':
                                for j in l:
                                    if res['arg']['instId'] == j['instrument_id']:
                                        # 获取全量数据
                                        bids_p = j['bids_p']
                                        asks_p = j['asks_p']
                                        # 获取合并后数据
                                        bids_p = update_bids(res, bids_p)
                                        asks_p = update_asks(res, asks_p)

                                        # 校验checksum
                                        checksum = res['data'][0]['checksum']
                                        # print('推送数据的checksum为：' + str(checksum))
                                        check_num = check(bids_p, asks_p)
                                        # print('校验后的checksum为：' + str(check_num))
                                        if check_num == checksum:
                                            print("校验结果为：True")
                                        else:
                                            print("校验结果为：False，正在重新订阅……")

                                            # 取消订阅
                                            await unsubscribe_without_login(url, channels)
                                            # 发送订阅
                                            async with websockets.connect(url) as ws:
                                                sub_param = {"op": "subscribe", "args": channels}
                                                sub_str = json.dumps(sub_param)
                                                await ws.send(sub_str)
                                                print(f"send: {sub_str}")
        except Exception as e:
            print("连接断开，正在重连……")
            continue

async def subscribe(url, api_key, passphrase, secret_key, channels):
    while True:
        try:
            async with websockets.connect(url) as ws:
                # login
                timestamp = str(get_local_timestamp())
                login_str = login_params(timestamp, api_key, passphrase, secret_key)
                await ws.send(login_str)
                # print(f"send: {login_str}")
                res = await ws.recv()
                print(res)

                # subscribe
                sub_param = {"op": "subscribe", "args": channels}
                sub_str = json.dumps(sub_param)
                await ws.send(sub_str)
                print(f"send: {sub_str}")
                cont = 0
                preco_pass = 0
                while True:
                    try:
                        res = await asyncio.wait_for(ws.recv(), timeout=25)
                    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
                        try:
                            marketDataAPI =  MarketData.MarketAPI(flag='1')
                            response = marketDataAPI.get_ticker(
                                instId='ETH-USDT'
                            )

                            preco_atual = response['data'][0]
                            print(preco_atual)
                            preco_atual = float(preco_atual['last'])
                            if preco_atual>preco_pass:
                                print('AJUSTE compra \n')
                                order_price = preco_atual*(1-0.002) #preço de compra
                                buy = 10/(float(order_price))

                                result = tradeAPI.amend_order(
                                    instId='ETH-USDT',
                                    newPx=str(order_price),
                                    newSz=str(buy),
                                    clOrdId = '100431ETHPPM'
                                )
                            preco_pass=preco_atual

                            await ws.send('ping')
                            res = await ws.recv()
                            print(res)
                            continue
                        except Exception as e:
                            print("12 连接关闭，正在重连……")
                            marketDataAPI =  MarketData.MarketAPI(flag='1')
                            response = marketDataAPI.get_ticker(
                                instId='BTC-USDT'
                            )
                            preco_atual = response['data'][0]
                            #print(preco_atual)
                            if float(preco_atual['last'])>preco_pass:
                                print(preco_atual)
                            preco_pass=float(preco_atual['last'])
                            break
                    if cont > 0:
                        print(get_timestamp() + res)
                        return res
                    else:
                        cont+=1
                        print(get_timestamp() + res)
                    

        except Exception as e:
            print("Desconectado, reconequitando")
            continue
# trade
async def trade(url, api_key, passphrase, secret_key, trade_param):
    while True:
        try:
            async with websockets.connect(url) as ws:
                # login
                timestamp = str(get_local_timestamp())
                login_str = login_params(timestamp, api_key, passphrase, secret_key)
                await ws.send(login_str)
                # print(f"send: {login_str}")
                res = await ws.recv()
                print(res)

                # trade
                sub_str = json.dumps(trade_param)
                await ws.send(sub_str)
                print(f"send: {sub_str}")

                while True:
                    try:
                        res = await asyncio.wait_for(ws.recv(), timeout=25)
                    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
                        try:
                            await ws.send('ping')
                            res = await ws.recv()
                            print(res)
                            continue
                        except Exception as e:
                            print("连接关闭，正在重连……")
                            break

                    print(get_timestamp() + res)
    
        except Exception as e:
            print("连接断开，正在重连……")
            continue
    


# unsubscribe channels
async def unsubscribe(url, api_key, passphrase, secret_key, channels):
    async with websockets.connect(url) as ws:
        # login
        timestamp = str(get_local_timestamp())
        login_str = login_params(timestamp, api_key, passphrase, secret_key)
        await ws.send(login_str)
        # print(f"send: {login_str}")

        res = await ws.recv()
        print(f"recv: {res}")

        # unsubscribe
        sub_param = {"op": "unsubscribe", "args": channels}
        sub_str = json.dumps(sub_param)
        await ws.send(sub_str)
        print(f"send: {sub_str}")

        res = await ws.recv()
        print(f"recv: {res}")


# unsubscribe channels
async def unsubscribe_without_login(url, channels):
    async with websockets.connect(url) as ws:
        # unsubscribe
        sub_param = {"op": "unsubscribe", "args": channels}
        sub_str = json.dumps(sub_param)
        await ws.send(sub_str)
        print(f"send: {sub_str}")

        res = await ws.recv()
        print(f"recv: {res}")


api_key = 'c526a733-e865-49bb-96f8-80efea44bc7b'
secret_key = 'B2E5C73BA075C7B03214B23F7C369BF4'
passphrase = 'Par@negociar1'

tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, '1')


# WebSocket公共频道 public channels
# 实盘 real trading
# url = "wss://ws.okx.com:8443/ws/v5/public"
# 模拟盘 demo trading
# url = "wss://ws.okx.com:8443/ws/v5/public?brokerId=9999"

# WebSocket私有频道 private channels
# 实盘 real trading
# url = "wss://ws.okx.com:8443/ws/v5/private"
# 模拟盘 demo trading
url = "wss://ws.okx.com:8443/ws/v5/private?brokerId=9999"

'''
公共频道 public channel
:param channel: 频道名
:param instType: 产品类型
:param instId: 产品ID
:param uly: 合约标的指数

'''

# Canal de produto
# channels = [{"channel": "instruments", "instType": "FUTURES"}]
# Canal de cotação
# channels = [{"channel": "tickers", "instId": "BTC-USD-210326"}]
# Canal de volume de posição
# channels = [{"channel": "open-interest", "instId": "BTC-USD-210326"}]
# Canal de K-line
# channels = [{"channel": "candle1m", "instId": "BTC-USD-210326"}]
# Canal de negociação
# channels = [{"channel": "trades", "instId": "BTC-USD"}]
# Canal de preço estimado de entrega/exercício
# channels = [{"channel": "estimated-price", "instType": "FUTURES", "uly": "BTC-USD"}]
# Canal de preço de marcação
# channels = [{"channel": "mark-price", "instId": "BTC-USDT-210326"}]
# Canal de K-line de preço de marcação
# channels = [{"channel": "mark-price-candle1D", "instId": "BTC-USD-201225"}]
# Canal de preço limite
# channels = [{"channel": "price-limit", "instId": "BTC-USD-201225"}]
# Canal de profundidade
# channels = [{"channel": "books", "instId": "BTC-USD-SWAP"}]
# Canal de precificação de opção
# channels = [{"channel": "opt-summary", "uly": "BTC-USD"}]
# Canal de taxa de financiamento
# channels = [{"channel": "funding-rate", "instId": "BTC-USD-SWAP"}]
# Canal de K-line de índice
# channels = [{"channel": "index-candle1m", "instId": "BTC-USDT"}]
# Canal de cotação de índice
# channels = [{"channel": "index-tickers", "instId": "BTC-USDT"}]
# Canal de status
# channels = [{"channel": "status"}]

'''
私有频道 private channel
:param channel: 频道名
:param ccy: 币种
:param instType: 产品类型
:param uly: 合约标的指数
:param instId: 产品ID

'''

# Canal de conta
# channels = [{"channel": "account", "ccy": "BTC"}]
# Canal de posições
# channels = [{"channel": "positions", "instType": "FUTURES", "instFamily": "BTC-USDT", "instId": "BTC-USDT-201225"}]
# Canal de ordens
channels = [{"channel": "orders", "instType": "SPOT"}]
# Canal de ordens de estratégia
# channels = [{"channel": "orders-algo", "instType": "FUTURES", "instFamily": "BTC-USD", "instId": "BTC-USD-201225"}]

'''
交易 trade
'''

# 下单
# trade_param = {"id": "1512", "op": "order", "args": [{"side": "buy", "instId": "BTC-USDT", "tdMode": "isolated", "ordType": "limit", "px": "19777", "sz": "1"}]}
# 批量下单
# trade_param = {"id": "1512", "op": "batch-orders", "args": [
#         {"side": "buy", "instId": "BTC-USDT", "tdMode": "isolated", "ordType": "limit", "px": "19666", "sz": "1"},
#         {"side": "buy", "instId": "BTC-USDT", "tdMode": "isolated", "ordType": "limit", "px": "19633", "sz": "1"}
#     ]}
# 撤单
# trade_param = {"id": "1512", "op": "cancel-order", "args": [{"instId": "BTC-USDT", "ordId": "259424589042823169"}]}
# 批量撤单
# trade_param = {"id": "1512", "op": "batch-cancel-orders", "args": [
#         {"instId": "BTC-USDT", "ordId": "259432098826694656"},
#         {"instId": "BTC-USDT", "ordId": "259432098826694658"}
#     ]}
# 改单
# trade_param = {"id": "1512", "op": "amend-order", "args": [{"instId": "BTC-USDT", "ordId": "259432767558135808", "newSz": "2"}]}
# 批量改单
# trade_param = {"id": "1512", "op": "batch-amend-orders", "args": [
#         {"instId": "BTC-USDT", "ordId": "259435442492289024", "newSz": "2"},
#         {"instId": "BTC-USDT", "ordId": "259435442496483328", "newSz": "3"}
#     ]}

response = {'data':[{'state':'cai'}]}
while response['data'][0]['state'] != 'canceled':

    loop = asyncio.get_event_loop()

    # 公共频道 不需要登录（行情，持仓总量，K线，标记价格，深度，资金费率等）
    # loop.run_until_complete(subscribe_without_login(url, channels))

    # 私有频道 需要登录（账户，持仓，订单等）
    response = loop.run_until_complete(subscribe(url, api_key, passphrase, secret_key, channels))

    # 交易（下单，撤单，改单等）
    # loop.run_until_complete(trade(url, api_key, passphrase, secret_key, trade_param))

    loop.close()

    response = json.loads(response)
    print('\n RESPOSTA \n', response)
    print('\n', response['data'])
