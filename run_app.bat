@echo off
:: Switch to the E: drive
E:

:: Navigate to the folder
cd "E:\BANA 275"

:: Run the Streamlit application
streamlit run paper_doc.py

:: Keep the window open if there is an error
pause