# AI-Trainer: 지역별 예상 숙박 가격 예측 시스템

## 프로젝트 개요
이 프로젝트는 호텔, 레저, 가격 데이터를 기반으로 숙박 가격을 예측하고 지도 및 시각화 기능을 제공하는 시스템입니다.
- **데이터 처리**: 원본 호텔 데이터에서 도로명주소를 가공하여 지역명을 추출합니다.
- **모델 학습**: 가격 데이터를 이용해 RandomForestRegressor 모델을 학습합니다.
- **예측 및 시각화**: Streamlit 앱을 통해 사용자 입력을 받고, 예측 결과와 간단한 시각화(막대 그래프)를 제공합니다.

## 프로젝트 구조
AI-Trainer/ 
├── data/ 
│ ├── raw/
│ │ ├── hotel_data.csv # 호텔 원본 데이터 
│ │ ├── leisure_data.csv # 레저 원본 데이터 
│ │ └── price.csv # 숙박 가격 원본 데이터 
│ └── processed/
│ └── processed_data.csv # 호텔 데이터 전처리 결과 (지역명 추가) 
│ ├── models/
│ ├── random_forest.pkl # 학습된 Random Forest 모델 
│ ├── encoder.pkl # OneHotEncoder 객체 
│ └── label_encoder.pkl # (필요 시) LabelEncoder 객체 
│ ├── assets/ │ └── fonts/ 
│ └── NanumBarunGothic.ttf # UI용 한글 글꼴 
│ ├── notebooks/
│ ├── encoding.ipynb # 인코딩 실험 노트북 
│ └── modeling.ipynb # 모델 학습 실험 노트북 
│ ├── data_preprocessing.py # 원본 호텔 데이터 가공 스크립트 
├── model_training.py # 가격 데이터 기반 모델 학습 스크립트
├── app.py # Streamlit 예측 및 시각화 앱 (메인 진입점) 
├── requirements.txt # 필요한 라이브러리 목록 
└── README.md # 프로젝트 개요 및 실행 안내


## 실행 방법
1. 필요한 라이브러리를 설치합니다:

pip install -r requirements.txt

markdown
복사
2. 원본 호텔 데이터를 가공합니다:
python data_preprocessing.py

markdown
복사
3. 모델 학습 스크립트를 실행하여 모델과 인코더를 생성하고 저장합니다:
python model_training.py

markdown
복사
4. Streamlit 앱을 실행합니다:
streamlit run app.py

perl
복사

## 기여 방법
GitHub 레포지토리를 fork 후, Pull Request로 기여할 수 있습니다.
