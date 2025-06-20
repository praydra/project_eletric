import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
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

# 2. 정규화
scaler = MinMaxScaler()
ts_scaled = scaler.fit_transform(daily_power.values.reshape(-1, 1))




# 3. 시퀀스 생성 함수
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 30
X, y = create_sequences(ts_scaled, SEQ_LEN)

# 4. 학습/테스트 분할
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 5. Tensor 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 6. LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1])
        return out

model = LSTMModel()

# 7. 학습 설정
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 300

# 8. 학습 루프
for epoch in range(EPOCHS):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 9. 예측 및 역정규화
model.eval()
with torch.no_grad():
    preds = model(X_test).numpy()

preds_rescaled = scaler.inverse_transform(preds)
y_test_rescaled = scaler.inverse_transform(y_test.numpy())

# 10. 시각화
plt.figure(figsize=(12, 5))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(preds_rescaled, label='LSTM Prediction', linestyle='--')
plt.legend()
plt.title('LSTM Forecast vs Actual')
plt.grid(True)
plt.show()

# 11. MSE 평가
mse = mean_squared_error(y_test_rescaled, preds_rescaled)
print(f"LSTM MSE: {mse:.4f}")
