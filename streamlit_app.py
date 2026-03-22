import streamlit as st
import traceback

st.set_page_config(page_title="Debug", layout="wide")

st.write("🚀 App starting...")

try:
    import src.dashboard_app as app
    st.write("✅ import success")

    try:
        app.main()
    except Exception:
        st.error("❌ main() 실행 중 에러")
        st.code(traceback.format_exc())

except Exception:
    st.error("❌ import 단계에서 에러 발생")
    st.code(traceback.format_exc())
