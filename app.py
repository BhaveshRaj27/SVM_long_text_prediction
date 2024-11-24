
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import streamlit as st
from utils import inference

if __name__ == "__main__":
    file_name = 'inference_doc.pdf'
    save_dir = './downloaded_inference_pdf'
    back_up = './backup_pdf'
    st.title("Classification the pdf")

    st.write("Provide a URL to a PDF file, and app will classify it.")

    url = st.text_input("Enter the URL of the PDF:")
    file_name = 'inference_doc.pdf'
    save_dir = './downloaded_inference_pdf'
    back_up = './backup_pdf'
    if st.button("Extract Text"):
        result =  inference(url,file_name,save_dir,back_up)
        if result == None:
            st.error("Please provide a valid PDF URL.")
        st.text_area("Predicted Class", result, height =300 )