# Импорт библиотек
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Установка токена аутентификации для доступа к API
authToken = 't.i3u0rK6kEv3SPH391u70UDxTCsUqaOBVzLOm6tWgijT_SQLV9GbEQljT41QvjnA5Hu7wu_CHxJSz1bd2IfaYsg'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {authToken}'
}

# Получение figi для дальнейшей работы
json_data = {
    "query": "Ozon",
    "instrumentKind": "INSTRUMENT_TYPE_UNSPECIFIED",
    "apiTradeAvailableFlag": True
}

response = requests.post(
    'https://invest-public-api.tinkoff.ru/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/FindInstrument',
    headers=headers,
    json=json_data
)

data = response.json()
if 'instruments' not in data or len(data['instruments']) == 0:
    raise Exception("Instrument not found. Please check the ticker name and try again.")

df = pd.json_normalize(data['instruments'])
print(df.head())  # Проверка содержимого DataFrame

if 'ticker' not in df.columns or 'figi' not in df.columns:
    raise Exception("Required columns are not in the DataFrame.")

# Используем тикер 'OZON'
if df[df['ticker'] == 'OZON'].empty:
    raise Exception("Ticker 'OZON' not found in the data. Available tickers: ", df['ticker'].unique())

figi = df[df['ticker'] == 'OZON']['figi'].values[0]

# Получение исторических данных о свечах
json_data = {
    "from": "2023-10-08T00:00:00.866Z",
    "to": "2023-11-08T00:00:00.866Z",
    "interval": "CANDLE_INTERVAL_DAY",
    "instrumentId": figi
}

response = requests.post(
    'https://invest-public-api.tinkoff.ru/rest/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles',
    headers=headers,
    json=json_data
)

data = response.json()
if 'candles' not in data or len(data['candles']) == 0:
    raise Exception("No candle data found for the specified period.")

candles = data['candles']
df = pd.json_normalize(candles)

# Преобразование времени в datetime
df['time'] = pd.to_datetime(df['time'])

# Предобработка данных
prices = []
for candle in candles:
    prices.append(float(candle['open']['units']) + float(candle['open']['nano']) / 1e9)

prices_array = np.array(prices[:-1])
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices_array.reshape(-1, 1))

# Создание набора данных для обучения
def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset)-time_steps):
        X.append(dataset[i:(i+time_steps), 0])
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)

time_steps = 12
X_train, y_train = create_dataset(prices_scaled, time_steps)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Создание модели нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(units=50, return_sequences=False),
    tf.keras.layers.Dense(units=1)
])

# Компиляция и обучение модели
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=1)

# Прогнозирование цен на следующие 12 дней
test_input = prices_scaled[-time_steps:].reshape((1, time_steps, 1))
predicted_prices_scaled = model.predict(test_input)
predicted_prices = scaler.inverse_transform(predicted_prices_scaled)

# Визуализация данных с помощью японских свечей
fig = go.Figure(data=[go.Candlestick(x=df['time'],
                open=df['open.units'],
                high=df['high.units'],
                low=df['low.units'],
                close=df['close.units'])])

fig.show()

# Построение графиков
start_date = datetime.strptime("2023-11-08T00:00:00.866Z", "%Y-%m-%dT%H:%M:%S.%fZ")
predicted_dates = [start_date + timedelta(days=i) for i in range(1, 13)]

plt.figure(figsize=(10, 5))
plt.plot(df['time'], prices, label='Истинные цены')
plt.plot(predicted_dates, predicted_prices.flatten(), label='Прогнозируемые цены', color='red')
plt.legend()
plt.show()
