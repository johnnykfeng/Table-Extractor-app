import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from TableExtraction import *
import os
from streamlit_download_button import download_button, file_selector


st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_page_config(layout='wide')

st.title("Table Extractor App :bookmark_tabs:")

# set up widgets to initialize settings for the table extractor
col1, col2, col3 = st.columns((1,1,1))
td_slider = col1.slider('Table Detection (TD) threshold', 0.0, 1.0, 0.5)
tsr_slider = col2.slider('Table Structure Recognition (TSR) threshold', 0.0, 1.0, 0.5)
crop_padding = col3.slider('Crop padding', 0, 60, 30)
ocr_choice = st.radio('Two different OCR can be used: Google is more accurate, Tesseract is faster.', 
                      ('Google-OCR', 'Tesseract'))



# if ocr_choice == 'Google-OCR':
#     api_key = st.text_input('Enter Google API key:')
# else:
#     api_key = None

# ocr_choice = 'Google-OCR'  # pytesseract doesn't work in Docker

# print(first_row_header_check)

# image uploader
img_name = st.file_uploader("Upload an image with table(s):")

if img_name is None:
    with st.container():
        sample_file = st.selectbox('Pre-loaded Samples:',
                                ('table3x3.jpg', 'hard_to_detect_tables.jpg','no_tables.jpg','salary_table.jpg', 
                                'three_tables.jpg', 'two_borderless_tables.jpg'))

        img_name = r'./samples/'+sample_file


# instantiate table extractor
te = TableExtractor()

# I ended up not using this
# @st.cache_data
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf-8')

# secret api key for google ocr
api_key = st.secrets["api_key"]
quota_project_id = st.secrets["quota_project_id"]

first_row_header_check = st.checkbox("First row header", value=False)

cell_img_dir = r"./cell_images" # directory for storing cropped cells
if img_name is not None:
    run_button = st.button('Run table extractor')

    if run_button:
        # try:
        if True:
            st.caption('running table extraction :hourglass:...')
            status_text = st.empty()
            
            df_list, TD_plot, TSR_plots = te.run_extraction(img_name, cell_img_dir = cell_img_dir, 
                                        TD_THRESHOLD=td_slider, 
                                        TSR_THRESHOLD=tsr_slider,
                                        ocr_choice=ocr_choice,
                                        api_key_string=api_key,
                                        quota_project_id=quota_project_id,
                                        print_progress=False, 
                                        show_plots=False, 
                                        first_row_header=first_row_header_check,
                                        delta_xmin=crop_padding, delta_ymin=crop_padding, delta_xmax=crop_padding, delta_ymax=crop_padding,
                                        status_text=status_text)
            
            if (df_list is not None): # check if any tables detected
                st.success('Table extraction finished!', icon="✅")
                
                with st.expander('Showing table feature detection:', expanded=True):
                    plt1, plt2 = st.columns([1,1], gap="medium")
                    plt1.subheader('Table Detection')
                    plt1.pyplot(TD_plot)
                    plt2.subheader('Table Structure Recognition')
                    
                    for plot in TSR_plots:
                        plt2.pyplot(plot)
        
                # delete all files in .\csv_files directory
                csv_dir = "./csv_files"
                if (not os.path.exists(csv_dir)): # creates folder if it doesn't exist
                    os.mkdir(csv_dir)
                for f in os.listdir(csv_dir):  # removes all existing files in the folder
                    os.remove(os.path.join(csv_dir, f))
                    
                st.subheader(':arrow_left: Download csv files in the next page')
                
                with st.expander('Show extracted dataframes:', expanded=True): 
                    for i, df in enumerate(df_list): # display each dataframe
                        st.subheader('Extracted Table #{}'.format(i+1))
                        st.dataframe(df) 
                        df.to_csv(f'./csv_files/datafile{i+1}.csv')
                        
                # st.subheader('**Download csv files** :arrow_down:')
                                                
        # except:
        #     st.error('**No tables detected!**')
           