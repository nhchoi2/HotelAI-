# File: app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 저장된 모델 및 인코더 로드
try:
    model = joblib.load("models/random_forest.pkl")
    encoder = joblib.load("models/encoder.pkl")
except Exception as e:
    st.error(f"모델 또는 인코더 로드 중 오류 발생: {e}")

st.title("🏨 지역별 예상 숙박 가격 예측")

# 사용자 입력 받기
regions = ["강원도 강릉시", "부산 해운대구", "제주 제주시"]
types = ["Hotel", "Pension", "Motel"]
seasons = ["성수기", "비수기"]

selected_region = st.selectbox("지역을 선택하세요", regions)
selected_type = st.selectbox("숙박 유형을 선택하세요", types)
selected_season = st.selectbox("성수기 여부를 선택하세요", seasons)

if st.button("예측하기"):
    try:
        # 입력 데이터를 DataFrame으로 생성
        input_data = pd.DataFrame([[selected_region, selected_type, selected_season]],
                                  columns=["지역명", "숙박유형명", "성수기여부"])
        
        # 성수기여부 숫자로 변환
        input_data["성수기여부"] = input_data["여행일자가 성수기에 해당하나요? 맞으면 1, 아니면 0을 선택해주세요"]
        
        # OneHotEncoder 적용: '지역명'과 '숙박유형명'
        X_transformed = encoder.transform(input_data[["지역명", "숙박유형명"]])
        
        # 최종 입력 데이터 결합: 인코딩된 결과와 성수기여부
        final_input = np.hstack([X_transformed, input_data[["성수기여부"]].values])
        
        # 예측 수행
        prediction = model.predict(final_input)[0]
        
        st.success(f"예상 숙박 가격: {int(prediction)} 원")
        
        # 간단한 시각화: 예측된 가격을 막대 그래프로 표시
        df_vis = pd.DataFrame({"항목": ["예상 가격"], "가격": [int(prediction)]})
        fig = px.bar(df_vis, x="항목", y="가격", title="예상 가격")
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")
