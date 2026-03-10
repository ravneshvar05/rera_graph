import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
<style>
    .llm-explanation {
        padding: 0 10px 15px 16px;
        border-left: 2px solid rgba(66, 153, 225, 1.0);
        margin-left: 10px;
        margin-bottom: 20px;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

st.title("Test Markdown rendering")

reasoning = """Here are some reasons:
- First reason
- Second reason
  - Nested reason
- Third reason

**Bold text** and *italic*."""

st.markdown(f'<div class="llm-explanation">\n\n{reasoning}\n\n</div>', unsafe_allow_html=True)
st.markdown("---")

st.markdown(f'<div class="llm-explanation" markdown="1">\n\n{reasoning}\n\n</div>', unsafe_allow_html=True)

