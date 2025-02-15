import folium
from folium.plugins import MarkerCluster
import pandas as pd

# 예시: 호텔, 레저 데이터프레임이 이미 로드되어 있다고 가정합니다.
# hotel_df: 숙박업이름, 숙박업유형명, 지역명, 위도, 경도, 객실이미지URL 등의 컬럼을 포함
# leisure_df: 레저상품명, 시설도로명주소, 시설위도, 시설경도 등의 컬럼을 포함

# 지도 초기화 (중심 좌표 [36.5, 127.5], zoom_start=10)
m = folium.Map(location=[36.5, 127.5], zoom_start=10)

# 호텔 데이터 클러스터 생성
hotel_cluster = MarkerCluster(name='호텔 데이터').add_to(m)

# 호텔 마커 추가: 마커를 클릭하면 팝업에 숙소 정보 표와 숙소 주변 레저 정보 리스트가 표시됨
for idx, row in hotel_df.iterrows():
    hotel_lat = row['위도']
    hotel_lon = row['경도']
    
    # 호텔 정보 표 HTML
    hotel_table = f"""
    <table border="1" style="border-collapse: collapse; width: 100%;">
      <tr><th>숙박업이름</th><td>{row['숙박업이름']}</td></tr>
      <tr><th>숙박업유형명</th><td>{row['숙박업유형명']}</td></tr>
      <tr><th>지역명</th><td>{row['지역명']}</td></tr>
    </table>
    """
    
    # 숙소 주변 레저 정보 (예: 거리 기준 0.01 정도 차이)
    nearby_leisure = leisure_df[
        (((leisure_df['시설위도'] - hotel_lat)**2 + (leisure_df['시설경도'] - hotel_lon)**2)**0.5) < 0.01
    ]
    
    leisure_list = "<ul>"
    if not nearby_leisure.empty:
        for _, leisure_row in nearby_leisure.iterrows():
            leisure_list += f"<li>{leisure_row['레저상품명']}<br>{leisure_row['시설도로명주소']}</li>"
        leisure_list += "</ul>"
    else:
        leisure_list = "<p>주변 레저 정보 없음</p>"
    
    popup_html = f"""
    {hotel_table}
    <br><b>주변 레저 시설:</b>
    {leisure_list}
    <br><img src="{row['객실이미지URL']}" width="200" alt="객실 이미지">
    """
    
    folium.Marker(
        location=[hotel_lat, hotel_lon],
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(color='blue', icon='home', prefix='fa')
    ).add_to(hotel_cluster)

# 레저 데이터 클러스터 (별도 표시)
leisure_cluster = MarkerCluster(name='레저 데이터').add_to(m)
for idx, row in leisure_df.iterrows():
    leisure_lat = row['시설위도']
    leisure_lon = row['시설경도']
    popup_html = f"""
    <b>레저상품명:</b> {row['레저상품명']}<br>
    <b>주소:</b> {row['시설도로명주소']}
    """
    folium.Marker(
        location=[leisure_lat, leisure_lon],
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(color='green', icon='star', prefix='fa')
    ).add_to(leisure_cluster)

# 가격 데이터는 별도 처리 (예: 지역별 대표 좌표 이용)
region_coords = hotel_df.groupby('지역명').agg({'위도':'mean', '경도':'mean'}).reset_index()
price_coords_df = pd.merge(price_df, region_coords, on='지역명', how='left')
price_coords_df_clean = price_coords_df.dropna(subset=['위도', '경도'])

price_cluster = MarkerCluster(name='가격 데이터').add_to(m)
for idx, row in price_coords_df_clean.iterrows():
    popup_html = f"""
    <b>숙박업유형명:</b> {row['숙박유형명']}<br>
    <b>평균판매금액:</b> {row['평균판매금액']}<br>
    <b>성수기여부:</b> {row['성수기여부']}
    """
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=7,
        color='red',
        fill=True,
        fill_color='red',
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(price_cluster)

# 레이어 컨트롤 추가
folium.LayerControl().add_to(m)

# HTML 파일로 저장 (실행 후 브라우저에서 확인 가능)
m.save("cluster_map_with_image.html")
