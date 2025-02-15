import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. 저장된 머신러닝 모델 및 인코더 로드
try:
    model = joblib.load("models/random_forest.pkl")    # Random Forest 모델 로드
    encoder = joblib.load("models/encoder.pkl")         # OneHotEncoder 객체 로드 (지역명, 숙박유형명 대상)
except Exception as e:
    st.error(f"모델 또는 인코더 로드 중 오류 발생: {e}")

# 2. Streamlit UI 설정
st.title("🏨 지역별 예상 숙박 가격 예측")

# 3. 사용자 입력 받기
regions = ["강원도 강릉시", "부산 해운대구", "제주 제주시"]
types = ["Hotel", "Pension", "Motel"]

# 성수기 여부를 숫자로 선택 (1: 성수기, 0: 비수기) - UI에서는 텍스트로 표시
selected_season = st.selectbox("성수기 여부를 선택하세요", [1, 0],
                               format_func=lambda x: "성수기" if x == 1 else "비수기")
selected_region = st.selectbox("지역을 선택하세요", regions)
selected_type = st.selectbox("숙박 유형을 선택하세요", types)

# 4. 예측 버튼 클릭 시 예측 수행
if st.button("예측하기"):
    try:
        # 입력 데이터를 DataFrame으로 생성 (세 컬럼 모두 포함)
        input_data = pd.DataFrame([[selected_region, selected_type, selected_season]], 
                                  columns=["지역명", "숙박유형명", "성수기여부"])
        # One-Hot Encoding 적용: '지역명'과 '숙박유형명'에 대해 인코딩
        X_transformed = encoder.transform(input_data[["지역명", "숙박유형명"]])
        # 만약 인코딩 결과가 sparse matrix이면 dense array로 변환
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        # 최종 입력 데이터 생성: 인코딩된 데이터와 '성수기여부'를 결합
        final_input = np.hstack([X_transformed, input_data[["성수기여부"]].values])
    
        # 예측 수행
        prediction = model.predict(final_input)[0]
        
        # 예측 결과 출력
        st.success(f"예상 숙박 가격: {int(prediction)} 원")

        # 데이터 로드 (CSV 파일에서 로드하는 예시)
        df_price = pd.read_csv("data/raw/price.csv")

        # 첫 번째 조건: "Hotel", "강원도 강릉시", 성수기(1)
        filtered_data_1 = df_price[(df_price['숙박유형명'] == 'Hotel') & 
                                (df_price['지역명'] == '강원도 강릉시') & 
                                (df_price['성수기여부'] == 1)]

        # 두 번째 조건: "Hotel", "강원도 강릉시", 비수기(0)
        filtered_data_2 = df_price[(df_price['숙박유형명'] == 'Hotel') & 
                                (df_price['지역명'] == '강원도 강릉시') & 
                                (df_price['성수기여부'] == 0)]

        # 성수기 통계값 계산
        stats_1 = {
            "최소값": filtered_data_1['평균판매금액'].min() if not filtered_data_1.empty else "데이터 없음",
            "최대값": filtered_data_1['평균판매금액'].max() if not filtered_data_1.empty else "데이터 없음",
            "평균값": int(filtered_data_1['평균판매금액'].mean()) if not filtered_data_1.empty else "데이터 없음"
        }

        # 비수기 통계값 계산
        stats_2 = {
            "최소값": filtered_data_2['평균판매금액'].min() if not filtered_data_2.empty else "데이터 없음",
            "최대값": filtered_data_2['평균판매금액'].max() if not filtered_data_2.empty else "데이터 없음",
            "평균값": int(filtered_data_2['평균판매금액'].mean()) if not filtered_data_2.empty else "데이터 없음"
        }

        # 첫 번째 DataFrame 생성 (성수기)
        df_1 = pd.DataFrame({
            "단위(원)": ["성수기"],
            "최저가": [stats_1["최소값"]],
            "최고가": [stats_1["최대값"]],
            "평균값": [stats_1["평균값"]]
        })

        # 두 번째 DataFrame 생성 (비수기)
        df_2 = pd.DataFrame({
            "단위(원)": ["비수기"],
            "최저가": [stats_2["최소값"]],
            "최고가": [stats_2["최대값"]],
            "평균값": [stats_2["평균값"]]
        })

        # 두 DataFrame 합치기
        result_df = pd.concat([df_1, df_2], ignore_index=True)

        # Streamlit에 결과 출력
        st.write("📋 성수기/비수기 가격 통계:")
        st.dataframe(result_df)


        fig = px.bar(df_prediction, 
            x="지역", 
            y=["성수기 가격 예측", "비수기 가격 예측", "선택된 지역 가격"],
            title=f"📊 {selected_region} 지역별 성수기/비수기 평균 가격 및 예상 가격",
            barmode="group")  # 그룹형 막대 그래프 설정

        # 그래프 간격을 조정하여 폭을 줄임
        fig.update_layout(
            bargap=0.15,  # 막대 간의 간격
            bargroupgap=0.1,  # 그룹 간의 간격
            autosize=True,  # 자동 크기 조정
            xaxis_title="지역",  # x축 라벨
            yaxis_title="가격(원)",  # y축 라벨
            plot_bgcolor="rgba(240, 240, 240, 1)",  # 그래프 배경색 (연한 회색)
            
            # 축 격자선 및 테두리 설정
            xaxis=dict(
                showgrid=True,  # x축에 격자선 표시
                zeroline=True,  # x축에서 0라인 표시
                showline=True,  # x축에 테두리 선 표시
                linecolor="black",  # x축 테두리 색상
            ),
            yaxis=dict(
                showgrid=True,  # y축에 격자선 표시
                zeroline=True,  # y축에서 0라인 표시
                showline=True,  # y축에 테두리 선 표시
                linecolor="black",  # y축 테두리 색상
            )
        )

        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")
