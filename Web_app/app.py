
import streamlit as vAR_st
import streamlit.components.v1 as components
import pandas as pd
import docx2txt
#from PIL import Image 
from PyPDF3 import PdfFileReader
import pdfplumber
import base64 
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

#for Setting the page layout to wide
vAR_st.set_page_config(layout="wide")

#for having the logo and title in the same line we use vAR.st_beta_columns() and make the ratio accordingly 
vAR_logo, vAR_title = vAR_st.columns((6,50))
with vAR_logo:
  vAR_st.image('https://raw.githubusercontent.com/DeepsphereAI/Streamlit-CommonToAllIndustry/master/Web_app/logo.jpg')
with vAR_title:
#setting font size and colour for the title 
  
#by this text-align: centre, we can align the title to the centre of the page
  vAR_st.markdown("<h1 style='text-align: center; color: green;'>Develop Enterprise-Grade Data Science Web Applications Using Streamlit</h1>", unsafe_allow_html=True)

#To customize the background colour of the submit button  
m = vAR_st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #f2f20a ;
}
</style>""", unsafe_allow_html=True)

components.html("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """)

def local_css(file_name):
    with open(file_name) as f:
        vAR_st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#local_css("style.css")
@vAR_st.cache(suppress_st_warning=True)
def csv_downloader(data):
  csvfile = data.to_csv()
  b64 = base64.b64encode(csvfile.encode()).decode()
  vAR_filename = "new_csv_file_{}_.csv".format(timestr)
  #vAR_st.markdown("#### Download File ###")
  href = f'<a href="data:file/csv;base64,{b64}" download="{vAR_filename}">Click Here!!</a>'
  vAR_st.markdown(href,unsafe_allow_html=True)

col1, col2, col3, col4, col5 = vAR_st.columns([0.5,1.5,2,0.5,1])
with col2:
    vAR_st.write('')
    vAR_st.write('')
    vAR_st.subheader("Problem Statement")
with col3:
    vAR_problem = vAR_st.selectbox('',('','why','when','what'),index=0)

col1, col2, col3, col4, col5 = vAR_st.columns([0.5,1.5,2,0.5,1])
with col2:
  if vAR_problem != '':
    vAR_st.write('')
    vAR_st.write('')
    vAR_st.subheader("Problem type")
with col3:
  if vAR_problem != '':
    vAR_type = vAR_st.selectbox('',('','Linear Regression','Nonlinear Regression','Bayesian Linear Regression'),index=0)

col1, col2, col3, col4, col5 = vAR_st.columns([0.5,1.5,2,0.5,1])
with col2:
  if vAR_problem != '':
    if vAR_type != '':
      vAR_st.write('')
      vAR_st.write('')
      vAR_st.subheader("Model Selection")
with col3:
  if vAR_problem != '':
    if vAR_type != '':
      vAR_model = vAR_st.selectbox('',('','Random train/test splits','Cross-Validation','Bootstrap'),index=0)

vAR_st.write('')
col1, col2, col3, col4, col5 = vAR_st.columns([0.5,1.5,2,0.5,1])
with col2:
  if vAR_problem != '':
    if vAR_type != '':
      if vAR_model != '':
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader("Training Dataset")
with col3:
  if vAR_problem != '':
    if vAR_type != '':
      if vAR_model != '':
        vAR_training_data = vAR_st.file_uploader("Upload CSV file")
with col5:
  if vAR_problem != '':
    if vAR_type != '':
      if vAR_model != '':
        validation = 'application/vnd.ms-excel'
        if vAR_training_data is not None:
          if vAR_training_data.type == validation:
            df_training = pd.read_csv(vAR_training_data, encoding = 'unicode_escape',error_bad_lines=False)
            vAR_st.markdown('#')
            vAR_st.write('')
            button_training = vAR_st.button('Preview', key="1")
            if button_training:
              csv_downloader(df_training)
          else:
            vAR_st.markdown('#')
            vAR_st.write('Upload CSV file you uploaded',vAR_training_data.type)

col1, col2, col3, col4, col5 = vAR_st.columns([0.5,1.5,2,0.5,1])
with col2:
  if vAR_problem != '':
    if vAR_type != '':
      if vAR_model != '':
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader("Testing Dataset")
with col3:
  if vAR_problem != '':
    if vAR_type != '':
      if vAR_model != '':
        vAR_testing_data = vAR_st.file_uploader("upload CSV file")
with col5:
  if vAR_problem != '':
    if vAR_type != '':
      if vAR_model != '':
        validation = 'application/vnd.ms-excel'
        if vAR_testing_data is not None:
          if vAR_testing_data.type == validation:
            df_testing = pd.read_csv(vAR_testing_data, encoding = 'unicode_escape',error_bad_lines=False)
            vAR_st.markdown('#')
            vAR_st.write('')
            button_testing = vAR_st.button('Preview', key="2")
            if button_testing:
              csv_downloader(df_testing)
          else:
            vAR_st.markdown('#')
            vAR_st.write('Upload CSV file you uploaded',vAR_testing_data.type)

col1, col2, col3, col4, col5 = vAR_st.columns([0.5,1.5,2,0.5,1])
with col2:
  if vAR_problem != '':
    if vAR_type != '':
      if vAR_model != '':
        if vAR_testing_data and vAR_training_data:
          vAR_st.write('')
          vAR_st.write('')
          vAR_st.subheader("Feature Engineering")
with col3:
  if vAR_problem != '':
    if vAR_type != '':
      if vAR_model != '':
        if vAR_testing_data and vAR_training_data:
          vAR_extraction = vAR_st.file_uploader("Feature Extraction")
with col5:
  if vAR_problem != '':
    if vAR_type != '':
      if vAR_model != '':
        if vAR_testing_data and vAR_training_data:
          validation = 'application/vnd.ms-excel'
          if vAR_extraction is not None:
            if vAR_extraction.type == validation:
              df_extraction = pd.read_csv(vAR_extraction, encoding = 'unicode_escape',error_bad_lines=False)
              vAR_st.markdown('#')
              vAR_st.write('')
              button_extraction = vAR_st.button('Preview', key="3")
              if button_extraction:
                csv_downloader(df_extraction)

            else:
              vAR_st.markdown('#')
              vAR_st.write('Upload CSV file you uploaded',vAR_extraction.type)    

col1, col2, col3, col4, col5 = vAR_st.columns([0.5,1.5,2,0.5,1])
with col2:
  if vAR_problem != '':
    if vAR_type != '':
      if vAR_model != '':
        if vAR_testing_data and vAR_training_data:
          if vAR_extraction:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.markdown('#')
            vAR_st.subheader("Model Engineering")
with col3:
  if vAR_problem != '':
    if vAR_type != '':
      if vAR_model != '':
        if vAR_testing_data and vAR_training_data:
          if vAR_extraction:
            vAR_train_model = vAR_st.selectbox('Training',('','Train Model'),index=0)
            vAR_st.write('')
            vAR_test_model = vAR_st.selectbox('Testing',('','Test Model'),index=0)
with col5:
  if vAR_problem != '':
    if vAR_type != '':
      if vAR_model != '':
        if vAR_testing_data and vAR_training_data:
          if vAR_extraction:
            if vAR_train_model == 'Train Model':
              vAR_st.write('')
              button_train_model = vAR_st.button('Preview', key="4")
              if button_train_model:
                csv_downloader(df_training)
            if vAR_test_model =='Test Model':
              vAR_st.markdown('#')
              vAR_st.write('')
              button_test_model = vAR_st.button('Preview', key="5")
              if button_test_model:
                csv_downloader(df_testing)
vAR_st.markdown("""---""")

menu = ["Home","Model Validation","Download Model Outcome","Data visualization","Deploy the Model"]
choice = vAR_st.sidebar.selectbox("Menu",menu)
if choice == "Home":
    vAR_st.subheader("Go to the Menu")

if choice == "Model Validation":
    vAR_st.subheader("Model Validation")
    vAR_st.button("Click here", key="6")

if choice == "Download Model Outcome":
    vAR_st.subheader("To Download the Model Outcome")
    vAR_st.button("Click here", key="7")

if choice == "Data visualization":
    vAR_st.subheader("Data visualization")
    vAR_st.button("Click here", key="8")

if choice == "Deploy the Model":
    vAR_st.subheader("To Deploy the Model")
    vAR_st.button("Click here", key="9")
