# File: data_preprocessing.py
import pandas as pd

# 원본 호텔 데이터 불러오기
hotel_df = pd.read_csv("data/raw/hotel_data.csv")

# "도로명주소"에서 앞의 두 단어를 추출하여 "지역명" 컬럼 생성
hotel_df["지역명"] = hotel_df["도로명주소"].str.split().str[:2].str.join(" ")

# 처리된 데이터 확인
print(hotel_df.head())

# 처리된 데이터를 processed_data.csv로 저장
hotel_df.to_csv("data/processed/processed_data.csv", index=False)
