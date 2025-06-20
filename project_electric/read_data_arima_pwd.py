import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
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


# 2. train-test 분할
train_size = int(len(daily_power) * 0.8)
train, test = daily_power[:train_size], daily_power[train_size:]
train = train.dropna() 
test = test.dropna()


# 3. auto_arima 모델 찾기 (계절성 없다고 가정)
stepwise_model = auto_arima(
    train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    seasonal=False,  # 비계절성
    d=None,          # 차분도 자동 결정
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print(f"\nBest model: {stepwise_model.summary()}")

# 4. 예측
n_periods = len(test)
forecast = stepwise_model.predict(n_periods=n_periods)

# 5. 성능 평가
mse = mean_squared_error(test, forecast)
print(f"Auto ARIMA MSE: {mse:.4f}")

# 6. 시각화
plt.figure(figsize=(10, 5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='Auto ARIMA Forecast', linestyle='--')
plt.legend()
plt.title('Auto ARIMA Forecast vs Actual')
plt.grid(True)
plt.show()
