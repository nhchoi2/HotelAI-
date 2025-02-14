import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# 📂 1. 저장된 머신러닝 모델 및 인코더 로드
try:
    model = joblib.load("models/random_forest.pkl")  # Random Forest 모델
    encoder = joblib.load("models/encoder.pkl")       # One-Hot Encoder
    label_encoder = joblib.load("models/label_encoder.pkl")  # Label Encoder (성수기여부)
except Exception as e:
    st.error(f"모델 또는 인코더 로드 중 오류 발생: {e}")

# 📂 2. Streamlit UI 설정
st.title("🏨 지역별 예상 숙박 가격 예측")

# 📂 3. 사용자 입력 받기
selected_region = st.selectbox("지역을 선택하세요", ["강원도 강릉시", "부산 해운대구", "제주 제주시"])
selected_type = st.selectbox("숙박 유형을 선택하세요", ["Hotel", "Pension", "Motel"])
selected_season = st.selectbox("성수기 여부를 선택하세요", ["성수기", "비수기"])

# 📂 4. 예측 버튼
if st.button("예측하기"):
    try:
        # 🔥 입력 데이터를 DataFrame으로 변환 (컬럼명에 공백이 없도록 처리)
        input_data = pd.DataFrame(
            [[selected_region, selected_type, selected_season]], 
            columns=["지역명", "숙박유형명", "성수기여부"]
        )
        # 혹시 모를 여분의 공백 제거
        input_data.columns = input_data.columns.str.strip()
        
        # 🔥 One-Hot Encoding 적용 (지역명, 숙박유형명)
        transformed_data = encoder.transform(input_data[["지역명", "숙박유형명"]])
        
        # 🔥 Label Encoding 적용 (성수기여부)
        # LabelEncoder는 1D 배열을 기대하므로 .values 사용
        transformed_season = label_encoder.transform(input_data["성수기여부"])
        print(transformed_season)
        input_data["성수기여부"] = transformed_season
        
        # 🔥 최종 입력 데이터 결합 (One-Hot Encoding 결과와 레이블 인코딩 결과 결합)
        final_input = np.hstack((transformed_data, input_data[["성수기여부"]]))
    
        # 🔥 예측 수행
        prediction = model.predict(final_input)[0]
        
        # 📂 5. 결과 출력
        st.success(f"📍 예상 숙박 가격: {int(prediction)} 원")
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")
    
    # 📂 6. 지역별 평균 가격 비교 차트
    df_prediction = pd.DataFrame({
        "지역": ["강원도 강릉시", "부산 해운대구", "제주 제주시"],
        "예상 가격": [120000, 98000, 110000]
    })
    fig = px.bar(df_prediction, x="지역", y="예상 가격", title="📊 지역별 예상 숙박 가격")
    st.plotly_chart(fig)
