import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px


model = joblib.load("models/random_forest.pkl")    # Random Forest 모델 로드
encoder = joblib.load("models/encoder.pkl")         # OneHotEncoder 객체 로드 (지역명, 숙박유형명 대상)

st.title("🏨 AI 숙박가격 예측 서비스")

tab1, tab2, tab3 = st.tabs(["앱 소개" , "예측", "내 주변 숙박&레져 정보"])

with tab1:
    st.header(" AI 기반 숙박 추천 & 지도 시각화 시스템 소개")
    st.markdown("""
    **AI-Trainer**는 사용자가 입력한 정보를 바탕으로 **머신러닝 모델을 활용하여 예상 숙박 가격을 예측**하고,  
    **지도 기반 시각화**를 통해 숙박 및 레저 정보를 한눈에 볼 수 있도록 지원하는 서비스입니다.  
    """)

    st.subheader("1. 프로젝트 개요")
    st.markdown("""
    - **프로젝트명:** AI-Trainer (AI 기반 숙박 추천 & 지도 시각화)
    - **목적:** 사용자의 숙박 유형, 지역, 성수기 여부 등을 입력하면 AI 모델이 예상 숙박 가격을 예측하고,  
      지도에서 호텔 및 주변 레저 시설 정보를 시각적으로 제공하는 시스템입니다.
    - **대상:** 여행객, 숙박 업계 관계자, 지역 관광 관련 종사자
    """)

    st.subheader("2. 데이터 출처")
    st.markdown("""
    - **숙박 및 레저 정보:** *야놀자 2024년 숙박 정보 및 레저 정보* → 정규화 후 데이터셋 적용  
    - **금액 정보:** *야놀자 무료 버전 (500개 지역별 평균 금액)* → 유료 데이터 접근 불가로 인해 무료 데이터셋 활용
    """)

    st.subheader("3. 주요 기능")
    st.markdown("""
     **AI 기반 숙박 가격 예측**  
    - 머신러닝(Random Forest) 모델을 활용하여 입력된 조건(지역, 숙박 유형, 성수기 여부)에 따라 **예상 숙박 가격을 제공**  
    - 사용자가 다양한 시나리오를 테스트하여 합리적인 가격을 찾을 수 있도록 지원  

     **지도 시각화 및 연관 정보 제공**  
    - **Folium 기반 인터랙티브 지도**를 통해 **호텔, 레저 시설, 지역별 평균 숙박 가격**을 한눈에 확인  
    - 숙소 주변의 **레저 시설 정보를 자동 연동**하여, 여행객이 쉽게 주변 시설을 탐색 가능  

     **사용자 친화적인 UI**  
    - **Streamlit을 활용한 직관적인 UI** 제공  
    - 간단한 입력값 설정만으로 AI가 최적의 숙박 정보를 추천  
    - **탭 방식의 네비게이션**으로 예측 결과와 지도 시각화를 쉽게 전환 가능  
    """)

    st.subheader("4. 시스템 구조")
    st.markdown("""
    **🛠 기술 스택 (Tech Stack)**  
    - **프론트엔드:** Streamlit (웹 UI)  
    - **백엔드:** Scikit-learn (Random Forest), Pandas, Joblib  
    - **데이터 시각화:** Plotly, Folium  
    - **데이터 저장:** CSV 기반 데이터 로드 (DB 미사용)  """)

    st.subheader("5. 기대 효과")
    st.markdown("""
    **차별점 (Why Our Solution?)**
    -  **AI 자동 가격 예측:** 단순 가격 비교가 아닌, 머신러닝 모델을 활용한 동적 예측  
    -  **숙박 & 레저 시설 연동:** 사용자가 원하는 숙소 주변의 여행지 정보를 자동 추천  
    -  **실시간 지도 기반 시각화:** 가격, 위치, 숙박 정보를 한눈에 비교 가능  

    **기대 효과**
    - 사용자 입장: 숙소 선택에 대한 신뢰성 증가 (예측된 가격과 실제 가격 비교 가능)  
    - 사업자 입장: 경쟁사 가격 데이터 분석 및 최적화 가능  
    - 관광 업계: 지역 내 숙박과 레저 활동을 연계하여 시너지 효과 창출  
    """)

    st.subheader("6.  배포 방법")
    st.markdown("""
    - 앱은 Streamlit을 사용하여 웹 애플리케이션 형태로 배포되었습니다.
    - 초기에는 로컬 환경에서 테스트 후, requirements.txt 파일을 생성하여 외부 환경에서도 실행 가능하도록 설정하였습니다.""")



with tab2:
    st.header("🏨 지역별 예상 숙박 가격 예측")
    # 예측 관련 UI 요소 배치


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
            filtered_data_1 = df_price[(df_price['숙박유형명'] == selected_type) & 
                                    (df_price['지역명'] == selected_region) & 
                                    (df_price['성수기여부'] == 1)]

            # 두 번째 조건: "Hotel", "강원도 강릉시", 비수기(0)
            filtered_data_2 = df_price[(df_price['숙박유형명'] == selected_type) & 
                                    (df_price['지역명'] == selected_region) & 
                                    (df_price['성수기여부'] == 0)]

            # 성수기 통계값 계산
            season_stats  = {
                "최소값": filtered_data_1['평균판매금액'].min(),
                "최대값": filtered_data_1['평균판매금액'].max(),
                "평균값": filtered_data_1['평균판매금액'].mean()
            }

            # 비수기 통계값 계산
            off_season_stats  = {
                "최소값": filtered_data_2['평균판매금액'].min(),
                "최대값": filtered_data_2['평균판매금액'].max(),
                "평균값": filtered_data_2['평균판매금액'].mean()
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


with tab3:
    st.markdown("""
    ### 내 주변 숙박 & 레저 시설 찾기
    - 현재 지도에서 **내 위치를 기준으로 숙박업소 및 레저 시설을 탐색**할 수 있습니다.
    - 숙박 시설 마커를 클릭하면 **해당 숙소의 상세 정보 및 주변 레저 활동을 확인**할 수 있습니다.
    - 지도 상의 클러스터를 활용하여 **지역별 숙박 및 레저 시설을 한눈에 파악**할 수 있습니다.
    """)
    from map_embed import embed_map
    # map_embed.py의 embed_map 함수를 호출하여 저장된 HTML 지도를 임베드

    embed_map(html_file="../notebooks/cluster_map_with_image.html", height=600)


    st.subheader("기능 안내")
    st.markdown("""
    - **숙박 정보 확인:** 호텔, 모텔, 펜션 등 숙박업소의 위치 및 정보를 지도에서 확인할 수 있습니다.
    - **레저 시설 탐색:** 숙소 주변의 레저 활동(테마파크, 체험 시설 등)을 연동하여 표시합니다.
    - **직관적인 인터페이스:** 클릭 한 번으로 숙소 및 주변 시설 정보를 빠르게 확인할 수 있습니다.
    """)
