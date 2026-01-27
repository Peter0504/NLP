@echo off
:: Change the drive to E:
E:
:: Change the directory to your project folder
cd "E:\BANA 275"
:: Run the Streamlit application
streamlit run doc_ui.py
:: Keep the window open if the app crashes
pause