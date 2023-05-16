import io
import os
import string
from collections import Counter
from itertools import count, tee
# import asyncio
from time import time
import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
# Torch only works for python 3.7 - 3.9, I learned this the hard way >.<!
import torch
from google.cloud import vision
from PIL import Image, ImageEnhance
from pytesseract import Output
# https://huggingface.co/docs/transformers/installation
from transformers import (DetrFeatureExtractor,
                          TableTransformerForObjectDetection)

def google_ocr_image_to_text(file_path, api_key_string, quota_project_id, CREDENTIALS=None):
    ''' Function that takes in an image file and returns the ocr text.
    Used in the last step of table extraction, on each of the cropped
    cell images. The cell images should be saved in a directory than input
    into this function sequentially. 
    
    Args:
        file_path: path for cell_images
        CREDENTIALS: sign up for google cloud platform and use your own
    '''
    
    client = vision.ImageAnnotatorClient(client_options={"api_key": api_key_string,
                                                        "quota_project_id": quota_project_id})
    
    with io.open(file_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)        
    response = client.document_text_detection(image=image)
    # return response.text_annotations
    if response.text_annotations == []:
        # print('empty cell!')
        return ''
    else:
        return response.text_annotations[0].description

# I have to install Tesseract-OCR in Windows and designate the path
# installation file is found here https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Feng\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def pytess(cell_pil_img):
    return ' '.join(pytesseract.image_to_data(cell_pil_img, output_type=Output.DICT, config='-c tessedit_char_blacklist=œ˜â€œï¬â™Ã©œ¢!|”?«“¥ --psm 6 preserve_interword_spaces')['text']).strip()
    
def uniquify(seq, suffs = count(1)):
    """Make all the items unique by adding a suffix (1, 2, etc).
    Credit: https://stackoverflow.com/questions/30650474/python-rename-duplicates-in-list-with-progressive-numbers-without-sorting-list
    `seq` is mutable sequence of strings.
    `suffs` is an optional alternative suffix iterable.
    """
    not_unique = [k for k,v in Counter(seq).items() if v>1] 

    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))  
    for idx, s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            continue
        else:
            seq[idx] += suffix

    return seq

def table_detector(image, THRESHOLD_PROBA):
    '''
    Table detection using pre-trained Detect-object TRansformer 

    '''

    feature_extractor = DetrFeatureExtractor(do_resize=True, size=800, max_size=800)
    encoding = feature_extractor(image, return_tensors="pt")

    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    with torch.no_grad():
        outputs = model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

    return (model, probas[keep], bboxes_scaled)

def table_struct_recog(image, THRESHOLD_PROBA):
    '''
    Table structure recognition using DEtect-object TRansformer pre-trained on 1 million tables
    '''

    feature_extractor = DetrFeatureExtractor(do_resize=True, size=1000, max_size=1000)
    encoding = feature_extractor(image, return_tensors="pt")

    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    with torch.no_grad():
        outputs = model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

    return (model, probas[keep], bboxes_scaled)

    '''
    Image padding as part of TSR pre-processing to prevent missing table edges
    '''
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    padded_image = Image.new(pil_img.mode, (new_width, new_height), color)
    padded_image.paste(pil_img, (left, top))
    return padded_image

class TableExtractor():

    colors = ["red", "blue", "green", "yellow", "orange", "violet"]

    def plot_TD(self, model, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        '''
        crop_tables and plot_TD must have same co-ord shifts because 1 only plots the other one updates co-ordinates 
        '''
        figTD, ax = plt.subplots(figsize=(16,10))
        # plt.figure(figsize=(16,10))
        ax.imshow(pil_img)
        # ax = plt.gca()

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cl = p.argmax()
            xmin_pad, ymin_pad, xmax_pad, ymax_pad = xmin-delta_xmin, ymin-delta_ymin, xmax+delta_xmax, ymax+delta_ymax 
            # red box shows model bounding box
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,fill=False, color='red', linewidth=1, linestyle='-.'))
            # blue line is after padding
            ax.add_patch(plt.Rectangle((xmin_pad, ymin_pad), xmax_pad - xmin_pad, ymax_pad - ymin_pad, fill=False, color='blue', linewidth=1, linestyle='--'))
            
            text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
            ax.text(xmin-20, ymin-50, text, fontsize=10,bbox=dict(facecolor='yellow', alpha=0.5))
            
        ax.axis('off')
        return figTD
        
    def crop_tables(self, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        '''
        crop_tables and plot_TD must have same co-ord shifts because 1 only plots the other one updates co-ordinates 
        '''
        cropped_img_list = []

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

            xmin, ymin, xmax, ymax = xmin-delta_xmin, ymin-delta_ymin, xmax+delta_xmax, ymax+delta_ymax 
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cropped_img_list.append(cropped_img)

        return cropped_img_list

    def generate_structure(self, model, pil_img, prob, boxes):
        """_summary_

        Args:
            model (_type_): ML model for TSR
            pil_img (_type_): cropped table image from TD
            prob (torch.tensor)): tensor with the scores for each label
            boxes (_type_): tensor with bounding box coordinates [xmin, ymin, xmax, ymax]

        Returns:
            _type_: _description_
        """
        
        '''
        Co-ordinates are adjusted here by 3 'pixels'
        To plot table pillow image and the TSR bounding boxes on the table
        '''
        rows = {}
        cols = {}
        idx = 0

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            
            # xmin, ymin, xmax, ymax = xmin, ymin, xmax, ymax  # is this line necessary?
            cl = p.argmax() # gets class dictionary 
            class_text = model.config.id2label[cl.item()]

            if class_text == 'table row':
                rows['table row.'+str(idx)] = (xmin, ymin, xmax, ymax)
            if class_text == 'table column':
                cols['table column.'+str(idx)] = (xmin, ymin, xmax, ymax)
            idx += 1
            
        return rows, cols
    
    def plot_TSR(self, model, pil_img, prob, boxes, tsr_fontsize=10):
        idx = 0  # loop index
        TSRfig = plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        plt.axis('off')
        
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cl = p.argmax() # gets class dictionary 
            class_text = model.config.id2label[cl.item()]
            
            if class_text == 'table row':  
                if idx % 2 == 0: # even
                    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=self.colors[cl.item()], linewidth=1, linestyle='--'))
                else: # odd
                    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=self.colors[cl.item()], linewidth=1, alpha=0.8))
                text = f'row: {p[cl]:0.2f}'
                ax.text(xmin-10, ymin-10, text, fontsize=tsr_fontsize, bbox=dict(facecolor=self.colors[cl.item()], alpha=0.2), ha='right', rotation=-30)
                
            if class_text == 'table column':
                if idx % 2 == 0: # even
                    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=self.colors[cl.item()], linewidth=1, linestyle='--'))
                else: # odd
                    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=self.colors[cl.item()], linewidth=1, alpha=0.8))
                text = f'column: {p[cl]:0.2f}'
                ax.text(xmin-10, ymin-10, text, fontsize=tsr_fontsize, bbox=dict(facecolor=self.colors[cl.item()], alpha=0.2), ha='left', rotation=30)  
        idx += 1 
        return TSRfig
        
    def sort_rows_cols(self, rows:dict, cols:dict):
        # Sometimes the header and first row overlap, and we need the header bbox not to have first row's bbox inside the headers bbox
        # sort rows by ymin
        rows_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(rows.items(), key=lambda tup: tup[1][1])}
        # sort cols by xmin
        cols_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(cols.items(), key=lambda tup: tup[1][0])}

        return rows_, cols_

    def crop_rows_cols(self, pil_img, rows:dict, cols:dict):
        """
        Extracts the box coordinates of the rows and cols, 
        then crops the rows and cell features into smaller images.

        Args:
        ----------
        pil_img : image file
            cropped table image from table_detector()
        rows, cols : 
            output from generate_structure --> sort_table_features

        Returns:
        -------
        rows, cols :
            added cropped_img as a value in each key
        """
        for k, v in rows.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax)) # crop the row
            rows[k] = xmin, ymin, xmax, ymax, cropped_img

        for k, v in cols.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax)) # crop the column
            cols[k] = xmin, ymin, xmax, ymax, cropped_img
            # plt.figure()
            # plt.imshow(cropped_img)
        return rows, cols

    def features_to_cells(self, master_row:dict, cols:dict):
        '''Removes redundant bbox for rows&columns and divides each row into cells from columns
        Args:

        Returns:
        '''
        cells_img = {}
        new_cols = cols
        new_master_row = master_row
        row_idx = 0
        for k_row, v_row in new_master_row.items():
            
            _, _, _, _, row_img = v_row
            xmax, ymax = row_img.size
            # cell_xmin, cell_ymin, cell_xmax, cell_ymax = 0, 0, 0, ymax
            x_offset = v_row[0]  # this variable fixed everything!
            row_img_list = [] # list for collecting cropped row_imgs
            # plt.imshow(row_img)
            for idx, kv in enumerate(new_cols.items()):
                k_col, v_col = kv
                xmin_col, _, xmax_col, _, col_img = v_col
                # xmin_col, xmax_col = xmin_col - padd_left - 10, xmax_col - padd_left
                xmin_col, xmax_col = xmin_col, xmax_col
                cell_xmin = xmin_col - x_offset
                cell_xmax = xmax_col - x_offset
                
                if idx == 0: # at the first column, crop at the left edge of row_img
                    cell_xmin = 0 
                if idx == len(new_cols)-1: # at the last column, crop at the right edge of row_img
                    cell_xmax = xmax
                    
                cell_xmin, cell_ymin, cell_xmax, cell_ymax = cell_xmin, 0, cell_xmax, ymax
                
                # pil_img.crop((xmin, ymin, xmax, ymax))
                row_img_cropped = row_img.crop((cell_xmin, cell_ymin, cell_xmax, cell_ymax)) # the actual cropping
                row_img_list.append(row_img_cropped)

            cells_img[k_row+'.'+str(row_idx)] = row_img_list
            row_idx += 1

        return cells_img, len(new_cols), len(new_master_row)-1

    def clean_dataframe(self, df):
        """ Remove irrelevant symbols that appear with OCR, this is most useful with pytesseract
        """
        for col in df.columns:
            df[col]=df[col].str.replace("|", '', regex=True)
            df[col]=df[col].str.replace("'", '', regex=True)
            df[col]=df[col].str.replace('"', '', regex=True)
            df[col]=df[col].str.replace(']', '', regex=True)
            df[col]=df[col].str.replace('[', '', regex=True)
            df[col]=df[col].str.replace('{', '', regex=True)
            df[col]=df[col].str.replace('}', '', regex=True)
        return df

    def save_df_to_csv(self, df, save_dir):
        return df.to_csv(save_dir, index=False)

    # def create_dataframe(self, c3, cells_pytess_result:list, max_cols:int, max_rows:int):
    def create_dataframe(self, ocr_text_list:list, max_rows:int, max_cols:int, first_row_header=False):
        '''Create dataframe using list of cell values of the table, also checks for valid header of dataframe
        Args:
            ocr_text_list (list of str): list of strings or coroutines, each element representing a cell in a table
            max_cols (int), max_rows (int): number of columns and rows
            first_row_header (bool): toggle first row header
        Returns:
            dataframe : final dataframe after all pre-processing 
        '''
        # print('text_list: ', ocr_text_list.shape)
        if first_row_header: # use first row as header
            headers = ocr_text_list[:max_cols] 
            print('HEADERS = {}'.format(headers))
            new_headers = uniquify(headers, (f' {x!s}' for x in string.ascii_lowercase))
            df = pd.DataFrame("", index=range(0, max_rows-1), columns=new_headers)
            
            cell_idx = max_cols # skip the first row
            for nrows in range(max_rows-1): 
                for ncols in range(max_cols):
                    # df.iat to access single cell values in a dataframe
                    # df.iat[nrows, ncols] = str(cells_list[cell_idx]) 
                    df.iat[nrows, ncols] = str(ocr_text_list[cell_idx]) 
                    cell_idx += 1
            
        else: # use numbers as header
            df = pd.DataFrame("", index=range(0, max_rows), columns=list(range(max_cols)))
            # df = pd.DataFrame("", index=range(0, max_rows))
        
            cell_idx = 0
            for nrows in range(max_rows):
                for ncols in range(max_cols):
                    # df.iat to access single cell values in a dataframe
                    # df.iat[nrows, ncols] = str(cells_list[cell_idx]) 
                    df.iat[nrows, ncols] = str(ocr_text_list[cell_idx]) 
                    cell_idx += 1
        
        return df

    # Runs the complete table extraction process of generating pandas dataframes from raw pdf-page images
    def run_extraction(self, image_path:str, cell_img_dir:str, api_key_string:str, quota_project_id:str,
                       TD_THRESHOLD=0.6, TSR_THRESHOLD=0.8, 
                       ocr_choice='Google-OCR',  print_progress=False, first_row_header=False, autosavecsv = False,
                       show_plots = True, tsr_fontsize=10,
                        delta_xmin=20, delta_ymin=20, delta_xmax=20, delta_ymax=20 , 
                        status_text=None, progress_bar=None):
        """Runs the complete table extraction process of generating pandas dataframes from raw pdf-page images

        Args:
            image_path (str): file path for input image
            cell_img_dir (_type_): directory path for storing cell images
            TD_THRESHOLD (float, optional): Defaults to 0.6.
            TSR_THRESHOLD (float, optional): Defaults to 0.8.
            use_pytess (bool, optional): set True if you want to use pytess Defaults to False.
            print_progress (bool, optional): Prints process for rows and cols. Defaults to False.
            first_row_header (bool, optional): Dedicate first row to header. Defaults to False.
            tsr_fontsize (int, optional): Fontsize for TSR labels. Defaults to 10.
            autosavecsv: Auto save df to csv. Defaults to False.
            show_plots: Plot results of TD and TSR. Recommend to be False when running multiple images. Defaults to True. 
            delta_xmin (int, optional): Pads xmin for cropping after TD. Defaults to 20.
            delta_ymin (int, optional): Pads ymin for cropping after TD. Defaults to 20.
            delta_xmax (int, optional): Pads xmax for cropping after TD. Defaults to 20.
            delta_ymax (int, optional): Pads ymax for cropping after TD. Defaults to 20.

        Returns:
            None: when no table detected in image
            df_list: list of extracted dataframes 
        """
        image = Image.open(image_path).convert("RGB")
        model, probas, bboxes_scaled = table_detector(image, THRESHOLD_PROBA=TD_THRESHOLD)

        if bboxes_scaled.nelement() == 0: # this ends the run_extraction() function 
            print('No table found in the pdf-page image')
            return None

        if show_plots:
            self.plot_TD(model, image, probas, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax) 

        TD_plot = self.plot_TD(model, image, probas, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax) 
        
        cropped_img_list = self.crop_tables(image, probas, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax)
        
        print('Number of tables in input image: {}'.format(len(cropped_img_list)))
        st.markdown(f'**{len(cropped_img_list)}** table(s) detected')
        
        print('Directory for storing cell images:', cell_img_dir)
        
        # checks for or creates the cell_img_dir 
        if (not os.path.exists(cell_img_dir)):
            os.mkdir(cell_img_dir)
        
        # clears directory of all previous cell images
        for f in os.listdir(cell_img_dir):
            os.remove(os.path.join(cell_img_dir, f))
        
        # loop over every image found by table_detector
        df_list = [] # list for collecting extracted dataframes
        TSR_plots = []
        for table in cropped_img_list:

            model, probas, bboxes_scaled = table_struct_recog(table, THRESHOLD_PROBA=TSR_THRESHOLD)
            
            rows, cols = self.generate_structure(model, table, probas, bboxes_scaled)
            
            if show_plots:
                self.plot_TSR(model, table, probas, bboxes_scaled, tsr_fontsize=tsr_fontsize)
            TSR_plot = self.plot_TSR(model, table, probas, bboxes_scaled, tsr_fontsize=tsr_fontsize)
            TSR_plots.append(TSR_plot)
            
            rows, cols = self.sort_rows_cols(rows, cols)
            
            master_row, cols = self.crop_rows_cols(table, rows, cols)

            cells_img, max_cols, max_rows = self.features_to_cells(master_row, cols)
                        
            n_cells = len(rows)*len(cols)
            table_info = f'Table dimensions: {len(rows)} rows, {len(cols)} columns, {n_cells} total cells'
            print(table_info)
            st.markdown(f'**{len(rows)}** rows x **{len(cols)}** columns = **{n_cells}** cells')
            cells_per_second_rate = 3.0
            # st.text(f'Estimated processing time: {n_cells/cells_per_second_rate :.1f} seconds')            
            
            if ocr_choice=='Google-OCR':
                google_ocr_texts = []  # initialize list for storing google ocr outputs
                # google_ocr_coroutines = []  # use this when applying async
                print('Start Google Vision OCR process...')
                start = time()
                j=0
                cell_counter=0
                prog_bar = st.progress(0)
                for k, img_list in cells_img.items():
                    if print_progress:
                        print('-- row: {} -'.format(j))

                    for i, img in enumerate(img_list):
                        if print_progress:
                            print('col: {}'.format(i))
                        
                        cell_counter += 1
                        if status_text is not None:  # output for streamlit app
                            status_text.caption('Processing cell #: {}'.format(cell_counter))
                            prog_bar.progress(cell_counter/n_cells)
                            
                        cell_name = '/cell_r{}_c{}.png'.format(j, i)
                        img.save(cell_img_dir + cell_name)
                        
                        # apply google ocr to the newly created cell image
                        google_ocr_texts.append(google_ocr_image_to_text(cell_img_dir + cell_name, 
                                                                         api_key_string = api_key_string, 
                                                                         quota_project_id = quota_project_id))
                        # text_google_ocr = google_ocr_image_to_text(cell_img_dir + cell_name, google_api_credentials)
                    j+=1        
                    
                # google_ocr_texts = await asyncio.gather(*google_ocr_coroutines)                        
                table_df = self.create_dataframe(google_ocr_texts,  len(rows), len(cols), first_row_header=first_row_header) # create dataframe
                # table_df = self.clean_dataframe(table_df)
                print("-- Google OCR Results --")
                print(table_df)
                if autosavecsv:
                    self.save_df_to_csv(table_df, 'table_df.csv')  
                print('Google OCR process took {:.2f} seconds'.format(time()-start)) 
                st.text('Processing time: {:.2f} seconds'.format(time()-start))
                 
            #-------------------------------------------------------------------------            
            if ocr_choice=='Tesseract':
                sequential_cell_img_list = [] # this is for pytess
                print('Start PyTesseract OCR process...')
                start = time()
                j=0
                cell_counter=0
                prog_bar = st.progress(0)
                for k, img_list in cells_img.items():
                    for i, img in enumerate(img_list):
                        sequential_cell_img_list.append(pytess(img))
                        cell_counter += 1
                        # progress_bar.progress(cell_counter/n_cells)
                        prog_bar.progress(cell_counter/n_cells)     
                    j+=1  
                
                table_df = self.create_dataframe(sequential_cell_img_list, len(rows), len(cols), first_row_header=first_row_header)
                # table_df = self.clean_dataframe(table_df)
                print("-- Pytesseract Results --")
                print(table_df)
                if autosavecsv:
                    self.save_df_to_csv(table_df, 'table_df.csv')    
            
                print('PyTesseract OCR process took {:.2f} seconds'.format(time()-start)) 
                st.text('Processing time: {:.2f} seconds'.format(time()-start))
                print('Errors in OCR is due to either quality of the image or performance of the OCR')
            #-----------------------------------------------------------------------------
            
            df_list.append(table_df)  
        
        if show_plots:
            # plt.ion()     
            plt.show() # I put this in the end to not interrupt the pipeline
        
        print('++-- END TABLE EXTRACTION --++')
        return df_list, TD_plot, TSR_plots
        
if __name__ == "__main__":
    
    cell_img_dir = r".\cell_images"
    # data_tables_dir = r"D:\_data_table_images"
    image_path = r'.\samples\table3x3.jpg'
    print('-- image_path --')
    print(image_path)
    
    TD_th = 0.95
    TSR_th = 0.8
    
    TD_padd = 30

    te = TableExtractor()
    if image_path is not None:
        te.run_extraction(image_path, cell_img_dir=cell_img_dir, TD_THRESHOLD=TD_th, TSR_THRESHOLD=TSR_th, 
                                     delta_xmin=TD_padd, delta_ymin=TD_padd, delta_xmax=TD_padd, delta_ymax=TD_padd)