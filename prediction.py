import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. 저장된 머신러닝 모델 및 인코더 로드
try:
    model = joblib.load("models/random_forest.pkl")  # 모델 로드
    encoder = joblib.load("models/encoder.pkl")  # One-Hot Encoding 변환기 로드
except Exception as e:
    st.error(f"모델 또는 인코더 로드 중 오류 발생: {e}")

# 2. Streamlit UI 설정
st.title("🏨 지역별 예상 숙박 가격 예측")

# 3. 사용자 입력 받기 (Select Box 사용)
city=['강원도 강릉시', '강원도 고성군', '강원도 동해시', '강원도 삼척시', '강원도 속초시', '강원도 양구군',
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
selected_region = st.selectbox("지역을 선택하세요", city)
selected_type = st.selectbox("숙박 유형을 선택하세요", ["Hotel", "Pension", "Motel"])
selected_season = st.selectbox("성수기 여부를 선택하세요", ["성수기", "비수기"])

# 선택된 값을 기반으로 숫자 반환
if selected_season == '성수기':
    season_value = 1  # 성수기인 경우 1
else:
    season_value = 0  # 비수기인 경우 0

# 이후 season_value 값을 사용하여 모델에 전달할 수 있습니다


# 4. 예측 버튼 추가
if st.button("예측하기"):
    try:
        # 입력 데이터를 DataFrame으로 변환
        input_data = pd.DataFrame([[selected_region, selected_type, selected_season]], 
                                  columns=["지역명", "숙박유형명", "성수기여부"])
        
        # 성수기/비수기 레이블 인코딩 (성수기 → 1, 비수기 → 0) - 직접 숫자로 변경
        input_data["성수기여부"] = input_data["성수기여부"].map({"성수기": 1, "비수기": 0})

        # 원핫 인코딩 적용
        categorical_columns = input_data[["지역명", "숙박유형명"]]
        X_transformed = encoder.transform(categorical_columns)

        # 최종 입력 데이터 생성
        final_input = np.hstack([X_transformed.toarray(), input_data[["성수기여부"]].values.reshape(-1, 1)])

        # 예측 수행
        prediction = model.predict(final_input)[0]

        # 결과 출력
        st.success(f"예상 숙박 가격: {int(prediction)} 원")

        df_price = pd.read_csv("data/price.csv")

        # 성수기
        filtered_data_1 = df_price[(df_price['숙박유형명'] == 'Hotel') & 
                           (df_price['지역명'] == '강원도 강릉시') & 
                           (df_price['성수기여부'] == 1)]
        
        # 비수기 
        filtered_data_2 = df_price[(df_price['숙박유형명'] == 'Hotel') & 
                           (df_price['지역명'] == '강원도 강릉시') & 
                           (df_price['성수기여부'] == 0)]        
        

        # 4. 성수기 및 비수기 가격의 최소값, 최대값, 평균값 계산
        season_stats = {
            "최소값": filtered_data_1['평균판매금액'].min(),
            "최대값": filtered_data_1['평균판매금액'].max(),
            "평균값": int(filtered_data_1['평균판매금액'].mean())
        }

        off_season_stats = {
            "최소값": filtered_data_2['평균판매금액'].min(),
            "최대값": filtered_data_2['평균판매금액'].max(),
            "평균값": int(filtered_data_2['평균판매금액'].mean())
        }

        # 5. 결과를 표로 출력
        stats_df = pd.DataFrame({
            "단위 (원)": ["성수기", "비수기"],
            "최소값": [season_stats["최소값"], off_season_stats["최소값"]],
            "최대값": [season_stats["최대값"], off_season_stats["최대값"]],
            "평균값": [season_stats["평균값"], off_season_stats["평균값"]],
            "예상가격": [[int(prediction)],[int(prediction)]]
        })

        st.write("📋 성수기/비수기 가격 통계:")
        st.dataframe(stats_df)

        # 6. 예측된 가격과 비교 그래프
        df_prediction = pd.DataFrame({
            "지역": [selected_region],
            "성수기 가격 예측": [season_stats["평균값"]],
            "비수기 가격 예측": [off_season_stats["평균값"]],
            "선택된 지역 가격": [prediction]  # 예측된 가격 (랜덤 포레스트 모델에서 가져온 예시)
        })


        fig = px.bar(df_prediction, 
             x="지역", 
             y=["성수기 가격 예측", "비수기 가격 예측", "선택된 지역 가격"],
             title=f"📊 {selected_region} 지역별 성수기/비수기 평균 가격 및 예상 가격",
             barmode="group")  # 그룹형 막대 그래프 설정
        # 그래프 간격을 조정하여 폭을 줄임
        fig.update_layout(
            bargap=0.10,  # 막대 간의 간격 (0과 1 사이로 설정, 작은 값일수록 간격 좁음)
            bargroupgap=0.5,  # 그룹 간의 간격
            autosize=True,  # 자동 크기 조정
            xaxis_title="지역",  # x축 라벨
            yaxis_title="가격(원)",  # y축 라벨
            plot_bgcolor="rgba(240, 240, 240, 1)",  # 그래프 배경색 (연한 회색)
    
        # 축 격자선 및 테두리 설정
            xaxis=dict(
                showgrid=True,  # x축에 격자선 표시
                zeroline=True,  # x축에서 0라인 표시
                showline=True,  # x축에 테두리 선 표시
                linecolor="black"  # x축 테두리 색상
            ),
            yaxis=dict(
                showgrid=True,  # y축에 격자선 표시
                zeroline=True,  # y축에서 0라인 표시
                showline=True,  # y축에 테두리 선 표시
                linecolor="black"  # y축 테두리 색상
            )
        )
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")
