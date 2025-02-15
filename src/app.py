import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import matplotlib as plt

# 1. 저장된 머신러닝 모델 및 인코더 로드
try:
    model = joblib.load("models/random_forest.pkl")    # Random Forest 모델 로드
    encoder = joblib.load("models/encoder.pkl")         # OneHotEncoder 객체 로드 (지역명, 숙박유형명 대상)
except Exception as e:
    st.error(f"모델 또는 인코더 로드 중 오류 발생: {e}")

tab1, tab2 = st.tabs(["예측", "지도 시각화"])

with tab1:
    st.header("예측 페이지")
    # 예측 관련 UI 요소 배치

with tab2:
    st.header("지도 시각화 페이지")

city = ['강원도 강릉시', '강원도 고성군', '강원도 동해시', '강원도 삼척시', '강원도 속초시', '강원도 양구군',
        '강원도 양양군', '강원도 영월군', '강원도 원주시', '강원도 인제군', '강원도 정선군', '강원도 철원군',
        '강원도 춘천시', '강원도 태백시', '강원도 평창군', '강원도 홍천군', '강원도 화천군', '강원도 횡성군',
        '경기도 가평군', '경기도 고양시덕양구', '경기도 고양시일산동구', '경기도 고양시일산서구', '경기도 과천시',
        '경기도 광명시', '경기도 광주시', '경기도 구리시', '경기도 군포시', '경기도 김포시', '경기도 남양주시',
        '경기도 동두천시', '경기도 부천시', '경기도 성남시분당구', '경기도 성남시수정구', '경기도 성남시중원구',
        '경기도 수원시권선구', '경기도 수원시영통구', '경기도 수원시장안구', '경기도 수원시팔달구', '경기도 시흥시',
        '경기도 안산시단원구', '경기도 안산시상록구', '경기도 안성시', '경기도 안양시동안구', '경기도 안양시만안구',
        '경기도 양주시', '경기도 양평군', '경기도 여주시', '경기도 연천군', '경기도 오산시',
        '경기도 용인시기흥구', '경기도 용인시수지구', '경기도 용인시처인구', '경기도 의왕시', '경기도 의정부시',
        '경기도 이천시', '경기도 파주시', '경기도 평택시']

# 2. Streamlit UI 설정
st.title("🏨 지역별 예상 숙박 가격 예측")

# 3. 사용자 입력 받기
regions = city
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
        season_stats  = {
            "최소값": filtered_data_1['평균판매금액'].min(),
            "최대값": filtered_data_1['평균판매금액'].max(),
            "평균값": int(filtered_data_1['평균판매금액'].mean())
        }

        # 비수기 통계값 계산
        off_season_stats  = {
            "최소값": filtered_data_2['평균판매금액'].min(),
            "최대값": filtered_data_2['평균판매금액'].max(),
            "평균값": int(filtered_data_2['평균판매금액'].mean())
        }

        stats_df = pd.DataFrame({
            "성수기/비수기": ["성수기", "비수기"],
            "최소값": [season_stats["최소값"], off_season_stats["최소값"]],
            "최대값": [season_stats["최대값"], off_season_stats["최대값"]],
            "평균값": [season_stats["평균값"], off_season_stats["평균값"]],
        })


        # Streamlit에 결과 출력
        st.write("📋 성수기/비수기 가격 통계:")
        st.dataframe(stats_df)


        df_prediction = pd.DataFrame({
            "지역": [selected_region],
            "성수기 가격 예측": [season_stats["평균값"]],
            "비수기 가격 예측": [off_season_stats["평균값"]],
            "선택된 지역 가격": [prediction]
        })

        fig = px.bar(df_prediction, 
             x="지역", 
             y=["성수기 가격 예측", "비수기 가격 예측", "선택된 지역 가격"],
             title=f"📊 {selected_region} 지역별 성수기/비수기 평균 가격 및 예상 가격",
             barmode="group")  # 그룹형 막대 그래프 설정
        

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
