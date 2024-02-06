# Bot
import pandas as pd
import numpy as np
from scipy.stats import norm
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
import time
from datetime import datetime
import matplotlib.pyplot as plt
import okx.MarketData as MarketData #Consultar o valor atual
import okx.Account as Account
import okx.Trade as Trade # Para negociar
import matplotlib.pyplot as plt

# Websocket
import asyncio
import websockets
import json
import requests
import hmac
import base64
import zlib
import webbrowser
import os

import httpx

api_key = 'c526a733-e865-49bb-96f8-80efea44bc7b'
secret_key = 'B2E5C73BA075C7B03214B23F7C369BF4'
passphrase = 'Par@negociar1'
flag = "1"  # live trading: 0, demo trading: 1
accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag) #Para acessar dados do fundo e configurações da conta
tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)

def calcular_indicadores_estatisticos(df):

    # Cálculo dos indicadores estatísticos
    ema = df['Close'].ewm(span=2, adjust=False).mean()  # Média Móvel Exponencial (EMA)
    delta = df['Close'].diff()  # Variação do preço
    gain = delta.where(delta > 0, 0)  # Ganho em cada período
    loss = -delta.where(delta < 0, 0)  # Perda em cada período
    avg_gain = gain.rolling(window=14).mean()  # Média móvel dos ganhos
    avg_loss = loss.rolling(window=14).mean()  # Média móvel das perdas
    rs = avg_gain / avg_loss  # Índice de Força Relativa (RSI)
    macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()  # Moving Average Convergence Divergence (MACD)

    # Incluir os indicadores estatísticos em uma cópia do dataframe original
    df_with_stats = df.copy()
    df_with_stats['EMA'] = ema
    df_with_stats['RSI'] = rs
    df_with_stats['MACD'] = macd

    # Tratamento dos dados
    df_with_stats.dropna(inplace=True)  # Remover valores ausentes
    df_with_stats = df_with_stats[~(df_with_stats == 0).any(axis=1)]  # Remover linhas com valores igual a zero
    df_with_stats = df_with_stats.loc[:, (df_with_stats != df_with_stats.iloc[0]).any()]  # Remover colunas com valores constantes    
    
    return df_with_stats


def classificar_oscilacao(df):
    
    # Ler o dataframe com as previsões de valor de fechamento
    df_previsao = df.copy() #Mudar aqui altera na avaliação das oscilações e quantidade de previsões disponíveis
    
    # Definir a janela de tempo em minutos
    janela = 2 # TENHO QUE MUDAR PARA QUE AS NEGOCIAÇÕES COMECEM A PARTIR DA SEGUNDA LINHA

    # Calcular a porcentagem de mudança do preço da moeda em cada janela de tempo
    df_previsao["Porcentagem"] = df_previsao["Close_prev"].pct_change(janela) * 100
    
    # Calcular a média e o desvio padrão das porcentagens de mudança em cada janela de tempo
    media = df_previsao["Porcentagem"].mean()
    desvio = df_previsao["Porcentagem"].std()

    # Usar a distribuição normal para estimar a probabilidade de ocorrência de uma determinada porcentagem de mudança em cada janela de tempo
    df_previsao["Probabilidade"] = norm.cdf(df_previsao["Porcentagem"], loc=media, scale=desvio)

    # Criar uma coluna com a direção da oscilação, usando o sinal da porcentagem de mudança
    df_previsao["Direcao"] = df_previsao.apply(lambda x: "Alta" if x['Porcentagem'] > 0 else "Baixa" if x['Porcentagem'] < 0 else "Estabilidade", axis=1)

    # Escolher um nível de significância para definir o limiar para qualificar as oscilações
    significancia = 0.05

    # Aplicar o limiar escolhido para classificar as oscilações como grandes ou pequenas em cada janela de tempo
    df_previsao["Oscilacao"] = df_previsao.apply(lambda x: "Grande" if x['Probabilidade'] < significancia or x['Probabilidade'] > 1 - significancia else "Pequena", axis=1)
    
    return df_previsao

model = load_model('BTCUSDT_2021-07-22_2023-12-27_1m.Tesv5.h5')

def previsao(instId):
    
    # Coletar dados históricos
    marketDataAPI =  MarketData.MarketAPI(flag=flag)

    # Retrieve the candlestick charts
    dados_atuais = marketDataAPI.get_candlesticks(
        instId=instId,
        bar='1m',
        limit = '300'
    )
    dados_atuais = dados_atuais['data']

    # Transforma a lista de dados em um dataframe
    df=pd.DataFrame(dados_atuais, columns=["Open time", "Open", "High", "Low", "Close","Volume", "VolCcy", "VolCcyQuote", "confirm"])

    # Selecionando apenas as colunas necessárias
    dados_iniciais = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Convertendo as colunas numéricas em float
    dados_iniciais.iloc[:, 1:] = dados_iniciais.iloc[:, 1:].astype(float)

#     dados_stat = calcular_indicadores_estatisticos(dados_iniciais)
    dados = calcular_indicadores_estatisticos(dados_iniciais)

    opentime = dados['Open time'].copy()
    opentime = opentime.reset_index(drop=True)

    # Separe os recursos (features)
    X = dados[['Open', 'High', 'Low','Close', 'Volume', 'EMA', 'MACD']]
    
    # Normalização dos recursos para valores entre 0 e 1
    scaler_input = StandardScaler()
    X_norm = scaler_input.fit_transform(X)

    # reshape input to be [samples, time steps, features]
    X_res = np.reshape(X_norm, (X_norm.shape[0], 1, X_norm.shape[1]))

    previsoes_norm = model.predict(X_res)

    X_desnorm = scaler_input.inverse_transform(X_norm) #dá pra tirar
    X_df = pd.DataFrame(data=X_desnorm, columns =[['Open', 'High', 'Low', 'Close_1m', 'Volume', 'EMA', 'MACD']] )
    X_df['Close_prev'] = previsoes_norm
    
    X_df.insert(loc=0, column='Open time', value=opentime) #Retornar a coluna Opentime
    
    #display(X_df)
#    inverter o dataframe, primeira linha vira a última
    X_df = X_df.iloc[::-1]
    X_df.reset_index(inplace=True, drop=True)
    
    df_previsao = classificar_oscilacao(X_df)

    df_previsao.to_csv(f'{instId}-Previsao_com_oscilacao.csv', index=False)

    return df_previsao

def ordem_compra(instId, px, sz, clOrdI):
    # Verificar se há uma condição de compra
    # Executar negociacao
    result = tradeAPI.place_order(
        instId=instId,
        tdMode="cash",
        side="buy",
        ordType="limit",
        px=str(px),
        sz=str(sz),
        clOrdId=clOrdI # you can define your own client defined order ID
    )
    
    
    return result

def ordem_venda(instId, px, sz, clOrdI): #

    result = tradeAPI.place_order(
        instId=instId,
        tdMode="cash",
        side="sell",
        ordType="limit",
        px=str(px),
        sz=str(sz),
        clOrdId=clOrdI # you can define your own client defined order ID
    )
    
    return result

def tentativa_compra(instId, px, sz, clOrdI):
    result = None
    # repete a chamada da função até que o resultado seja "0" ou o tempo limite seja atingido
    while result is None or result["code"] != "0":
        # chama a função comando_negociacao
        result = ordem_compra(instId, px, sz, clOrdI)
        # verifica se o resultado foi "0"
        if result["code"] == "0":
            # imprime a mensagem de sucesso
            print("Successful order request，order_id = ",result["data"][0]["clOrdId"])
        else:
            # imprime a mensagem de erro
            print("Unsuccessful order request，error_code = ",result["data"][0]["sCode"], ", Error_message = ", result["data"][0]["sMsg"])
        time.sleep(1)
    return result

def tentativa_venda(instId, px, sz, clOrdI):
    result = None
    # repete a chamada da função até que o resultado seja "0" ou o tempo limite seja atingido
    while result is None or result["code"] != "0":
        # chama a função comando_negociacao
        result = ordem_venda(instId, px, sz, clOrdI)
        # verifica se o resultado foi "0"
        if result["code"] == "0":
            # imprime a mensagem de sucesso
            print("Successful order request，order_id = ",result["data"][0]["clOrdId"])
        else:
            # imprime a mensagem de erro
            print("Unsuccessful order request，error_code = ",result["data"][0]["sCode"], ", Error_message = ", result["data"][0]["sMsg"])
        time.sleep(1)
    return result


def prev_direcao(df_previsao, coluna):

    prev = df_previsao[coluna].tail(1)
    
    #Resetar os indexadores para não dá problema 
    prev.reset_index(inplace=True, drop=True)

    prev = prev.iloc[0,0]

    return prev

def coleta_registros(df_previsao, coluna):

    prev = df_previsao[coluna].tail(1)
    
    #Resetar os indexadores para não dá problema 
    prev.reset_index(inplace=True, drop=True)

    prev = prev.iloc[0]

    return prev

def grafico_realprevisao(df_previsao):

    # Plotar os preços reais e previstos
    plt.figure(figsize=(20, 6))
    plt.plot(df_previsao.loc[:,'Close_1m'], label='Variacao real')
    #criar um eixo y secundário
    plt.twinx()

    plt.plot(df_previsao.loc[:, 'Close_prev'], label='Variacao prevista', color='red')
    plt.legend()
    # Adicionar a função plt.grid()
    plt.grid(True)
    
    # Salvando o gráfico em um arquivo
    plt.savefig("Real x Previsao.png")

    return

def calcular_valorizacao(df_resultado):
    # Crie uma coluna 'Valorizacao_real' com valores padrão como 0
    df_resultado['Valorizacao_real'] = 0
    df_resultado['Valorizacao_prev'] = 0

    # Calcule a valorização com base na condição Pt+h > Pt
    for i in range(len(df_resultado)-1):
        if df_resultado.iloc[i+1]['Close_1m'] > df_resultado.iloc[i]['Close_1m']:
            df_resultado.at[i, 'Valorizacao_real'] = 1
        if df_resultado.iloc[i+1]['Close_prev'] > df_resultado.iloc[i]['Close_prev']:
            df_resultado.at[i, 'Valorizacao_prev'] = 1
            
    df_resultado.dropna(inplace=True)

    return df_resultado

def avaliar_previsoes(df_resultado):
    df_resultado = calcular_valorizacao(df_resultado)

    # Calcule a acurácia, sensibilidade (recall) e AUC
    accuracy = accuracy_score(df_resultado['Valorizacao_real'], df_resultado['Valorizacao_prev'])
    recall = recall_score(df_resultado['Valorizacao_real'], df_resultado['Valorizacao_prev'])
    roc_auc = roc_auc_score(df_resultado['Valorizacao_real'], df_resultado['Valorizacao_prev'])

    # Calcule a matriz de confusão
    cm = confusion_matrix(df_resultado['Valorizacao_real'], df_resultado['Valorizacao_prev'])

    # Extraia os valores da matriz de confusão
    TN = cm[0][0]
    FP = cm[0][1]

    # Calcule a especificidade
    especificidade = TN / (TN + FP)

    # Ler o arquivo csv e armazenar em uma variável
    #df_original = pd.read_csv('Report_de_desempenho_das_previsoes.csv')

    #linha = len(df_original)
    #df_original.at[linha,'Time'] = time.time()
    #df_original.at[linha,'Acuracia_(%)'] = round(accuracy * 100, 2)
    #df_original.at[linha,'Sensibilidade_(%)'] = round(recall * 100, 2)
    #df_original.at[linha,'Especificidade_(%)'] = round(especificidade * 100, 2)
    #df_original.at[linha,'AUC'] = round(roc_auc, 2)

    # Salvar o novo dataframe em um arquivo csv
    #df_original.to_csv('Report_de_desempenho_das_previsoes.csv', index=False)

    print(f'Acurácia do modelo: {accuracy * 100:.2f}%')
    print(f'Sensibilidade (Recall): {recall * 100:.2f}%')
    print(f'Especificidade: {especificidade * 100:.2f}%')
    print(f'AUC (Área sob a Curva ROC): {roc_auc:.4f}')

# Chamando a função de atualização a cada 10 segundos

def bot(par, USDT, desvalorizacao, valorizacao ): 

    # Inicializar as variáveis de controle
    compra = True # Indica se há uma operação de compra em andamento
    lucro = 0 # Lucro 
    lucro_acumulado = 1
    contagem = 0 # Número de operações realizadas
    registros = [] # Lista de registros das operações
    preco_venda_real = 0
    preco_compra = 0
    preco_venda = 0
    crip_real = 0
    valor_inicial = USDT

    #Parâmetros e variáveis para previsao
    operacao_alta = False #Quando for realizado uma compra esperando alta grande
    valorizacao_alta = valorizacao*2
    desvalorizacao_baixa = 0.02

    #registros_pass = pd.read_csv(f'Registros {par[:3]}.csv')
    #lucro_acumulado = coleta_registros(registros_pass, 'Lucro Acum.(%)')
    #USDT = coleta_registros(registros_pass, 'USDT')

    #operacao = coleta_registros(registros_pass, 'Operação')
    #if operacao == "Venda":
        #compra = False

    api_key = 'c526a733-e865-49bb-96f8-80efea44bc7b'
    secret_key = 'B2E5C73BA075C7B03214B23F7C369BF4'
    passphrase = 'Par@negociar1'

    livro_retorno = []

    while True:

        #if lucro_acumulado >= 6.0:
         #   retorno = USDT - valor_inicial
          #  livro_retorno.append({"Retorno": retorno})
           # df_livro_retorno = pd.DataFrame(livro_retorno)
            #df_livro_retorno.to_csv(f'Retorno {par[:3]}.csv')
            #USDT = valor_inicial
         
        df_previsao = previsao(par)

        # pega a data e hora atual em UTC
        inicio = datetime.utcnow()
        
        previsao_direcao = prev_direcao(df_previsao, 'Direcao')
        previsao_oscilacao = prev_direcao(df_previsao, 'Oscilacao')
        grafico_realprevisao(df_previsao)
        avaliar_previsoes(df_previsao)

        #Consultar taxas de maker (comprador) e taker (vendedor)
        taxa = accountAPI.get_fee_rates(
            instType="SPOT",
            instId=par
        )    
        fee_m = float(taxa['data'][0]['maker'])

        #Consultar valor atual
        marketDataAPI =  MarketData.MarketAPI(flag=flag)
        response = marketDataAPI.get_ticker(
            instId=par
        )
        print(response)

        # ID da ordem. Terminacao m é compra (maker)
        clOrdId = str(int(time.time()))+par[:3]+'MM4'

        if compra:
            
            if previsao_direcao == 'Baixa' and previsao_oscilacao == 'Grande':

                desvalorizacao_usada = desvalorizacao_baixa
                valorizacao_usada = valorizacao_alta

                tipo_venda = "Venda na alta"
                tipo_compra = "Compra na baixa"

   
            elif previsao_direcao == 'Alta' and previsao_oscilacao == 'Grande':

                desvalorizacao_usada = desvalorizacao
                valorizacao_usada = valorizacao_alta

                tipo_venda = "Venda na alta"
                tipo_compra = "Compra"

            else: 
                desvalorizacao_usada = desvalorizacao
                valorizacao_usada = valorizacao

                tipo_venda = "Venda"
                tipo_compra = "Compra"

            order_price = float(response['data'][0]['last'])*(1-desvalorizacao_usada) #preço de compra
            crip = USDT/(float(order_price)) #quantidade de BTC
            print('COMPRAR')
            
            tentativa_compra(par, order_price, crip, clOrdId)        
            #operacao_alta = True

            hora_conclusao = datetime.utcnow()
            # Registrar a operação de compra
            registros.append({"Data": hora_conclusao, "clordID": clOrdId, "Operação": tipo_compra, "Status": 'Aberto', "Preço de compra": preco_compra, "Preço de venda esperado": preco_venda, "Preço venda real": preco_venda_real, "USDT": USDT, f"{par}": crip_real, "Lucro (%)": lucro, "Lucro Acum.(%)": lucro_acumulado})

            # Criar um dataframe com os registros das operações
            df_registros = pd.DataFrame(registros)
            df_registros.to_csv(f'Registros {par[:3]}.csv')

            trade = verificar_execucao(api_key, passphrase, secret_key, compra, par, clOrdId, order_price, desvalorizacao_usada, USDT, valorizacao_usada, crip_real, lucro_acumulado)
            compra = False
            print(trade)      

            #Verificar a que preço foi comprado
            preco_compra = float(trade['avgPx'])
            preco_venda = preco_compra*(1+valorizacao_usada)    

            # O quanto tenho da moeda ao descontar as taxas
            crip = float(trade['sz'])  
            USDT = crip*preco_compra*(1-fee_m)
            crip_real = crip*(1+fee_m)
            
            hora_conclusao = datetime.utcnow()
            # Registrar a operação de compra
            registros.append({"Data": hora_conclusao, "clordID": clOrdId, "Operação": tipo_compra, "Status": 'Fechado', "Preço de compra": preco_compra, "Preço de venda esperado": preco_venda, "Preço venda real": preco_venda_real, "USDT": USDT, f"{par}": crip_real, "Lucro (%)": lucro, "Lucro Acum.(%)": lucro_acumulado})
        
        contagem += 1

        # Criar um dataframe com os registros das operações
        df_registros = pd.DataFrame(registros)
        df_registros.to_csv(f'Registros {par[:3]}.csv')

        clOrdId = str(int(time.time()))+par[:3]+'MT4' #final um indica que é o primeiro programa 
        
        df_previsao = previsao(par)
        previsao_direcao = prev_direcao(df_previsao, 'Direcao')
        previsao_oscilacao = prev_direcao(df_previsao, 'Oscilacao')
        grafico_realprevisao(df_previsao)
        avaliar_previsoes(df_previsao)

        if not compra:
                
            #operacao_alta = False
            #print('VENDER NA ALTA \n')
            tentativa_venda(par, preco_venda, crip_real, clOrdId) 

            hora_conclusao = datetime.utcnow()
            # Registrar a operação de compra
            registros.append({"Data": hora_conclusao, "clordID": clOrdId, "Operação": tipo_venda, "Status": 'Aberto',"Preço de compra": preco_compra, "Preço de venda esperado": preco_venda, "Preço venda real": preco_venda_real, "USDT": USDT, f"{par}": crip_real, "Lucro (%)": lucro, "Lucro Acum.(%)": lucro_acumulado})
            # Criar um dataframe com os registros das operações
            df_registros = pd.DataFrame(registros)
            df_registros.to_csv(f'Registros {par[:3]}.csv')
            
            trade = verificar_execucao(api_key, passphrase, secret_key, compra, par, clOrdId, preco_compra, desvalorizacao_usada, USDT, valorizacao_usada, crip_real, lucro_acumulado)
            compra = True
            
            preco_venda_real = float(trade['avgPx'])
            print(preco_venda_real)
            qtd_vendida = float(trade['sz'])
            print(qtd_vendida)
            USDT_final = preco_venda_real*qtd_vendida
            print(USDT_final, USDT)
            lucro = (USDT_final-USDT)/USDT*100
            lucro_acumulado = (USDT_final - valor_inicial)/valor_inicial*100
            USDT = USDT_final
            crip_real = crip_real-qtd_vendida
            
            hora_conclusao = datetime.utcnow()
            # Registrar a operação de compra
            registros.append({"Data": hora_conclusao, "clordID": clOrdId, "Operação": tipo_venda, "Status": 'Fechado',"Preço de compra": preco_compra, "Preço de venda esperado": preco_venda, "Preço venda real": preco_venda_real, "USDT": USDT, f"{par}": crip_real, "Lucro (%)": lucro, "Lucro Acum.(%)": lucro_acumulado})

        # Criar um dataframe com os registros das operações
        df_registros = pd.DataFrame(registros)
        df_registros.to_csv(f'Registros {par[:3]}.csv')

        print("RESULTADO")
        print(df_registros)

        # Exibir o número de operações realizadas e o lucro final
        print(f"Número de operações realizadas: {contagem}")
        print(f"Lucro final: {lucro_acumulado:.2f} %")
        fim = datetime.utcnow()
        #inicio_data_hora = inicio.strtime(formato)
        print('Inicio:',inicio, '\nFim:', fim, '\nDuração:', fim-inicio)


def get_timestamp():
    now = datetime.now()
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

def segunda_verificacao(tradeAPI, instId, clOrdId):
    response = None
    # repete a chamada da função até que o resultado seja "0" ou o tempo limite seja atingido
    while response is None or response["code"] != "0":
        # chama a função comando_negociacao
        response = tradeAPI.get_order(
            instId=instId,
            clOrdId=clOrdId
        )
        # verifica se o resultado foi "0"
        if response["code"] == "0":
            continue
            # imprime a mensagem de sucesso
            #print("Successful order request，order_id = ",response["data"][0]["clOrdId"])
        else:
            # imprime a mensagem de erro
            print(response)
        time.sleep(1)
    return response


def ajuste(tradeAPI, instId, order_price, buy, clOrdId):
    
    result = tradeAPI.amend_order(
        instId=instId,
        newPx=str(order_price),
        newSz=str(buy),
        clOrdId=clOrdId
    )
    return result

async def subscribe(url, channels, api_key, passphrase, secret_key, 
                    compra, instId, clOrdId, 
                    valor_ordem, desvalorizacao, USDT, 
                    valorizacao, sz, lucro_acum
                    ):
    
    time_start = int(clOrdId[:-6])
    time_startback = time.time()
    trocou = False

    dados = []

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
                
                while True:
                    try:
                        res = await asyncio.wait_for(ws.recv(), timeout=25)
                        
                    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
                        
                        try:
                            await ws.send('ping')
                            await ws.recv()

                            tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
                            marketDataAPI =  MarketData.MarketAPI(flag='1')
                            response = marketDataAPI.get_ticker(
                                instId=instId
                            )
                            preco_atual = float(response['data'][0]['last'])
                            
                            dif_max = desvalorizacao*1.8
                            # Ajustar sl de acordo com o lucro acumulado
                            sl = valorizacao*1.5 #diferença decimal entre valor presente e valor de venda enviado, caso venda, estarei a dois ciclos de operacao de recuperar
                            perda_lucro = 0.8 
                            # Diferença de preço entre o valor venda desejado e o valor de atual
                            dif_preco = (valor_ordem-preco_atual)/preco_atual
                            #print(time.time()>time_start+600, time.time(), time.time()>time_start+600)
  
                            if compra and abs(dif_preco)>dif_max and time.time()>time_start+600:
                                #print('AJUSTE compra \n')
                                order_price = preco_atual*(1-desvalorizacao) #preço de compra
                                buy = USDT/(float(order_price))
                                #print(order_price>valor_ordem, order_price, valor_ordem)
                                if order_price>valor_ordem:
                                    #print(f'Ajuste compra: {result}')
                                    res = ajuste(tradeAPI, instId, order_price, buy, clOrdId)
                                    print(res)

                                    if res['data'][0]['sMsg'] == 'Your order has already been filled or canceled':
                                        res = segunda_verificacao(tradeAPI, instId,clOrdId)
                                        return res['data'][0]

                                time_start = time.time()

                            # Se tantos porcentos do lucro(perda_lucro) forem maior que a perda gerada pela desvalorização (deveria ser valorização, pois é o ganho que esperava ter) da moeda, então pode vender e resultar em uma perda do lucro acumulado
                            # criar um dicionário com as variáveis e seus nomes
                            dados.append({
                                "Operacao": clOrdId,
                                "dif_preco": dif_preco,
                                "sl": sl,
                                "lucro_perda": (lucro_acum*perda_lucro)/100,
                            })

                            # criar um dataframe com o dicionário
                            df = pd.DataFrame(dados)

                            # salvar o dataframe em um arquivo csv
                            df.to_csv(f"Venda_antecipada_{par}.csv", index=False)

                            if not compra and dif_preco>sl and (lucro_acum*perda_lucro)/100>dif_preco:
                                print(valor_ordem, preco_atual)
                                #print('AJUSTE venda \n')
                                res = ajuste(tradeAPI, instId, preco_atual, sz, clOrdId)
                                print(res)
                                
                                if res['data'][0]['sMsg'] == 'Your order has already been filled or canceled':
                                    res = segunda_verificacao(tradeAPI, instId,clOrdId)
                                    return res['data'][0]                     


                            #print('aqui')
                            #print(time.time()>time_startback+40)
                            if time.time()>time_startback+40: # 300 é 5 miin
                                res = segunda_verificacao(tradeAPI, instId,clOrdId)
                                time_startback = time.time()
                                #print("Segunda verificação")
                                print(res)
                                return res['data'][0]
                            
                        except IndexError:
                            if res['msg'] == 'API endpoint request timeout ':
                                if trocou == False: 
                                    api_key = 'd1c64656-e404-42d7-a962-c850df6035f2'
                                    secret_key = '320DB9091EB16DB2C718BC53A1F5C0F9'
                                    passphrase = 'Segundafrase@1'
                                    trocou = True
                                else: 
                                    api_key = 'c526a733-e865-49bb-96f8-80efea44bc7b'
                                    secret_key = 'B2E5C73BA075C7B03214B23F7C369BF4'
                                    passphrase = 'Par@negociar1'
                                    trocou = False
                            continue                            

                    try:
                        if not res == 'pong':
                            print(res)
                            res = json.loads(res)
                            #print(res)
                            res['data'][0]
                            return res['data'][0]
                    except KeyError:
                        print('Esperando')
                        continue
                    except TypeError:
                        print('Erro de tipo')
                        continue
                    except IndexError:
                        if res['msg'] == 'API endpoint request timeout ':
                            if trocou == False: 
                                api_key = 'd1c64656-e404-42d7-a962-c850df6035f2'
                                secret_key = '320DB9091EB16DB2C718BC53A1F5C0F9'
                                passphrase = 'Segundafrase@1'
                                trocou = True
                            else: 
                                api_key = 'c526a733-e865-49bb-96f8-80efea44bc7b'
                                secret_key = 'B2E5C73BA075C7B03214B23F7C369BF4'
                                passphrase = 'Par@negociar1'
                                trocou = False
                        continue

        except (websockets.exceptions.ConnectionClosed, httpx.ReadTimeout, websockets.exceptions.InvalidMessage, TimeoutError, ConnectionResetError, httpx.ConnectTimeout) as e:            
            continue
        except IndexError:
            if res['msg'] == 'API endpoint request timeout ':
                if trocou == False: 
                    api_key = 'd1c64656-e404-42d7-a962-c850df6035f2'
                    secret_key = '320DB9091EB16DB2C718BC53A1F5C0F9'
                    passphrase = 'Segundafrase@1'
                    trocou = True
                else: 
                    api_key = 'c526a733-e865-49bb-96f8-80efea44bc7b'
                    secret_key = 'B2E5C73BA075C7B03214B23F7C369BF4'
                    passphrase = 'Par@negociar1'
                    trocou = False
            continue

def verificar_execucao(api_key, passphrase, secret_key, compra, instId, clOrdId, valor_ordem, desvalorizacao, USDT, valorizacao, sz, lucro_acum):
    
    url = "wss://ws.okx.com:8443/ws/v5/private?brokerId=9999"

    # Canal de ordens
    channels = [{"channel": "orders", "instType": "SPOT"}]

    #response = {'data':[{'state':'demo', 'clOrdId':'demo'}]}

    pendente = True

    while pendente:

        loop = asyncio.get_event_loop()

        response = loop.run_until_complete(subscribe(url, channels, api_key, passphrase, secret_key, compra, instId, clOrdId, valor_ordem, desvalorizacao, USDT, valorizacao, sz, lucro_acum))
        #print(response)
        #try:
            #print('\n', response)
        #except IndexError:
            #print('Erro')
            #print(response)
            #response = segunda_verificacao(instId,clOrdId) 
            
        print("Fim ciclo \n", response)

        if response['state'] == 'filled' and response['clOrdId'] == clOrdId:
            pendente = False

        if response['state'] == 'canceled' and response['clOrdId'] == clOrdId:
            return 
        
        
    return response

USDT = 100
desvalorizacao = 0.003
valorizacao = 0.004

par = 'SUI-USDT'

print('USDT', USDT, '\n Desvalorização', desvalorizacao, '\n Valorização', valorizacao,'\n')

#df_desempenho = pd.DataFrame(columns=["Time","Acuracia_(%)","Sensibilidade_(%)","Especificidade_(%)", "AUC"], index=None)
#df_desempenho.to_csv(f'{par}-Report_de_desempenho_das_previsoes.csv', index=False)

bot(par, USDT, desvalorizacao, valorizacao )
