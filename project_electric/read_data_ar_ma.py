import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# 데이터 파일 경로 설정
file_path = 'household_power_consumption.txt'
# 데이터 로드
df = pd.read_csv(
    file_path,
    sep=';',  # 세미콜론으로 구분된 텍스트 파일
    parse_dates={'Datetime': ['Date', 'Time']},
    infer_datetime_format=True,
    na_values='?',
    low_memory=False
)

# 데이터의 shape 확인
print(df.shape)

# 데이터의 head 확인
print(df.head())

# 전력 소비 컬럼을 float으로 변환
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

# 결측치 제거
df = df.dropna()

# 시간 인덱스 설정
df.set_index('Datetime', inplace=True)

# 1일 단위로 리샘플링 
daily_power = df['Global_active_power'].resample('D').mean()


# 3. 학습/테스트 분리
train_size = int(len(daily_power) * 0.8)
train, test = daily_power[:train_size], daily_power[train_size:]


#- 분리 후, 혹시 남아있을 수 있는 NaN이나 무한대(inf) 값들을 확인하고 제거
print(train.isna().sum())  # NaN 값 개수 확인
print(np.isinf(train).sum())  # inf 값 개수 확인

train = train.dropna()  
test = test.dropna()

# 4. AR 모델 학습 및 예측
# - AutoReg: 과거 10일의 데이터를 사용해 현재를 예측하는 AR(10) 모델을 학습
ar_model = AutoReg(train, lags=10).fit()
ar_pred = ar_model.predict(start=len(train), end=len(daily_power)-1, dynamic=False)

# 5. MA 모델 학습 및 예측 (ARIMA(0,q,0)으로 구현)
ma_model = ARIMA(train, order=(0, 0, 10)).fit()
ma_pred = ma_model.forecast(steps=len(test))


plt.figure(figsize=(12, 5))
plt.plot(test.index, test, label='Actual', color='black')

print(test.index.shape)  # x축 크기 확인
print(ar_pred.shape)  # y축 크기 확인
print(ma_pred.shape)  # y축 크기 확인

ar_pred = ar_pred[:len(test.index)]
ma_pred = ma_pred[:len(test.index)]
plt.figure(figsize=(12, 5))
plt.plot(test.index, ar_pred, label='AR(10) Prediction', linestyle='--')

# 6. 결과 시각화
plt.figure(figsize=(12, 5))
plt.plot(test.index, test, label='Actual', color='black')
plt.plot(test.index, ar_pred, label='AR(10) Prediction', linestyle='--')
plt.plot(test.index, ma_pred, label='MA(10) Prediction', linestyle=':')
plt.legend()
plt.title('AR vs MA Forecast')
plt.ylabel('Power Consumption')
plt.grid(True)
plt.show()

# 7. 성능 평가
ar_mse = mean_squared_error(test, ar_pred)
ma_mse = mean_squared_error(test, ma_pred)

print(f"AR(10) MSE: {ar_mse:.4f}")
print(f"MA(10) MSE: {ma_mse:.4f}")