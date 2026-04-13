import os
from pathlib import Path
import streamlit as st
 
st.write("Python:", os.sys.version)
st.write("CWD:", os.getcwd())
 
candidates = [
    Path("data"),
    Path("data/raw"),
    Path("data/processed"),
    Path("data/exports"),
    Path("data/processed/analysis_results.csv"),
]
 
for p in candidates:
    st.write(str(p), "exists =", p.exists(), "resolved =", p.resolve())
    if p.exists() and p.is_file():
        st.write("size =", p.stat().st_size)

from src.dashboard_app import main

if __name__ == "__main__":
    main()
