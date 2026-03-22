import traceback
import streamlit as st

st.set_page_config(page_title="Debug Mode", layout="wide")

try:
    from src.dashboard_app import main
    main()

except Exception as e:
    st.error("❌ 앱 시작 중 에러 발생")
    st.code(traceback.format_exc())
