import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Data

#배터리 잔량 Data
acity = np.array([0, 1000, 3000, 5000, 7000, 10000, 13000, 16000]).reshape(-1, 1)
#충전 시간 Data
charging_time = np.array([2.5, 3.5, 3.0, 2.8, 3.2, 3.8, 3.3, 2.9])
#온도 Data
temperature = np.array([2, 4, 6, 10, 12, 17, 19, 21]).reshape(-1, 1)

# 데이터셋 분리
X_train = np.concatenate((battery_capacity, temperature), axis=1)
X_test = np.array([[4000, 5]])  # Test용 Data
y_train = charging_time

# 선형 회귀 모델 생성 및 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# Data Receive
user_battery_capacity = float(input("배터리 잔량: "))
user_temperature = float(input("현재 온도: "))

# Data 형식 변환
user_input = np.array([[user_battery_capacity, user_temperature]])

# 모델 가동
user_charging_time_pred = model.predict(user_input)

# 결과 출력
print(f"현재 상태에서 완충까지 남은 시간은 {user_charging_time_pred[0]:.2f} 시간입니다.")