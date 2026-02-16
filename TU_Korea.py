import numpy as np
from sklearn.linear_model import LinearRegression

# 배터리 잔량 Data
battery_capacity = np.array(
    [0, 1000, 3000, 5000, 7000, 10000, 13000, 16000]
).reshape(-1, 1)

# 충전 시간 Data (시간)
charging_time = np.array([2.5, 3.5, 3.0, 2.8, 3.2, 3.8, 3.3, 2.9])

# 온도 Data
temperature = np.array(
    [2, 4, 6, 10, 12, 17, 19, 21]
).reshape(-1, 1)


# 모델 학습
X_train = np.concatenate((battery_capacity, temperature), axis=1)
y_train = charging_time

model = LinearRegression()
model.fit(X_train, y_train)

user_battery_capacity = float(input("배터리 잔량을 입력하세요: "))
user_temperature = float(input("현재 온도를 입력하세요: "))

user_input = np.array([[user_battery_capacity, user_temperature]])
pred_time = model.predict(user_input)

print(f"완충까지 예상 시간: {pred_time[0]:.2f} 시간")
