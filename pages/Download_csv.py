import streamlit as st
from streamlit_download_button import download_button, file_selector

st.set_page_config(page_title='Download csv here', page_icon='📁', layout="centered", initial_sidebar_state="auto")

filename = file_selector(folder_path="./csv_files")

# Load selected file
with open(filename, 'rb') as f:
    s = f.read()
    
# download_button(s, filename, f'Click here to download {filename}')
download_button_str = download_button(s, filename, f'Click here to download {filename}')
st.markdown(download_button_str, unsafe_allow_html=True)