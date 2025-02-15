# File: model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

# 가격 데이터 불러오기
price_df = pd.read_csv("data/raw/price.csv")

# X: '지역명', '숙박유형명', '성수기여부'; y: '평균판매금액'
X = price_df[["지역명", "숙박유형명", "성수기여부"]].copy()
y = price_df["평균판매금액"]

# '성수기여부'를 숫자로 변환 (성수기: 1, 비수기: 0)
X["성수기여부"] = X["성수기여부"].map({"성수기": 1, "비수기": 0})

# OneHotEncoder: '지역명'과 '숙박유형명'에 적용
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X[["지역명", "숙박유형명"]])

# 최종 입력 데이터: 인코딩된 데이터와 '성수기여부' 컬럼 결합
X_final = np.hstack([X_encoded, X[["성수기여부"]].values])

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# RandomForestRegressor 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 간단한 평가 출력
score = model.score(X_test, y_test)
print(f"모델 R^2 점수: {score:.3f}")

# 모델 및 인코더 저장
joblib.dump(model, "models/random_forest.pkl")
joblib.dump(encoder, "models/encoder.pkl")

print("모델 학습 완료 및 저장되었습니다.")
