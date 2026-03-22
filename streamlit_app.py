import traceback
import streamlit as st

try:
    from src.dashboard_app import main
except Exception:
    def main():
        st.set_page_config(page_title="Startup error", layout="wide")
        st.error("앱 시작 중 오류가 발생했습니다.")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
