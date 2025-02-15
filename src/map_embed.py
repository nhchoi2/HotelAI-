# File: src/map_embed.py
import os
import streamlit as st

def embed_map(html_file="cluster_map_with_image.html", height=600):
    """
    저장된 HTML 파일(예: Folium 지도 파일)을 동적으로 경로를 생성하여
    Streamlit 앱에 임베드합니다.
    
    Parameters:
        html_file (str): 임베드할 HTML 파일명 (기본값: "cluster_map_with_image.html")
        height (int): 임베드 영역의 높이 (픽셀 단위, 기본값: 600)
    """
    # map_embed.py 파일이 있는 src 폴더 기준으로 notebooks 폴더의 경로 생성
    current_dir = os.path.dirname(__file__)  # src 폴더의 경로
    file_path = os.path.join(current_dir, "..", "notebooks", html_file)
    
    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        st.error(f"파일이 존재하지 않습니다: {file_path}")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=height)
    except Exception as e:
        st.error(f"지도를 임베드하는 중 오류 발생: {e}")
