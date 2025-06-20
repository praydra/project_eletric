import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

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
daily_power = daily_power.dropna()


# 2. 정상성 확인 (ADF 테스트)
result = adfuller(daily_power)
print(f"Before differencing - ADF p-value: {result[1]:.4f}")

# 3. 차분 (1차 차분)
ts_diff = daily_power.diff().dropna()

# 4. 다시 ADF 테스트
result_diff = adfuller(ts_diff)
print(f"After differencing - ADF p-value: {result_diff[1]:.4f}")

# 5. 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
daily_power.plot(title='Original Series')
plt.grid(True)
plt.subplot(1, 2, 2)
ts_diff.plot(title='1st-order Differenced Series')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. 학습/테스트 분할
train_size = int(len(daily_power) * 0.8)
train, test = daily_power[:train_size], daily_power[train_size:]
train = train.dropna() 
test = test.dropna()

# 7. ARIMA 모델 학습 (ARIMA(p,d,q) → 여기서는 ARIMA(5,1,2) 사용 예)
model = ARIMA(train, order=(5, 1, 2))  # (p=5, d=1, q=2)
model_fit = model.fit()

# 8. 예측
forecast = model_fit.forecast(steps=len(test))

# 9. 성능 평가
mse = mean_squared_error(test, forecast)
print(f"ARIMA(5,1,2) MSE: {mse:.4f}")

# 10. 시각화
plt.figure(figsize=(10, 5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='ARIMA Forecast', linestyle='--')
plt.legend()
plt.title('ARIMA(5,1,2) Forecast vs Actual')
plt.grid(True)
plt.show()
