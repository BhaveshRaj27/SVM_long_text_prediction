import requests
import os
import pandas as pd
import wget
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC

train_folder = './downloaded_train_pdfs'
backup_folder = './downloaded_pdfs'

def download_pdf_wget(url, save_dir, file_name):
    """
    Download a PDF using wget and save it locally.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        
        file_path = os.path.join(save_dir, file_name)
        wget.download(url, file_path)
        print(f"\nDownloaded: {url} -> {file_path}")
        return file_path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None
    
def download_pdf_requests(url, save_dir, file_name):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return None

# Function to extract text from PDFs using LangChain
def extract_text_with_langchain(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        # Combine text from all pages
        return " ".join([doc.page_content for doc in documents])
    except Exception as e:
        print(f"Failed to extract text from {file_path}: {e}")
        return None

# Function to extract target from PDF filename
def extract_target_from_filename(filename):
    try:
        # Extract target as the last word separated by '_'
        target = filename.split('_')[-1].split('.')[0]
        return target
    except Exception as e:
        print(f"Failed to extract target from filename {filename}: {e}")
        return None

# Function to generate backup filename for searching in backup folder
def generate_backup_filename(filename):
    try:
        # Extract document_x from document_x_name.pdf
        parts = filename.split('_')
        backup_filename = f"{parts[0]}_{parts[1]}.pdf" if len(parts) > 2 else filename
        return backup_filename
    except Exception as e:
        print(f"Failed to generate backup filename for {filename}: {e}")
        return None

# Function to attempt extraction from backup folder if primary fails
def extract_text_with_backup(filename,train_folder,backup_folder):
    primary_path = os.path.join(train_folder, filename)
    backup_filename = generate_backup_filename(filename)
    backup_path = os.path.join(backup_folder, backup_filename)

    # Attempt primary extraction
    text = extract_text_with_langchain(primary_path)
    if text is not None:
        return text

    # Attempt backup extraction
    print(f"Trying backup folder for {backup_filename}...")
    text = extract_text_with_langchain(backup_path)
    if text is not None:
        return text

    # If both fail, return None
    print(f"Extraction failed for {filename} in both folders.")
    return None

def preprocess_text(text, remove_stopwords=True, use_stemming=False, use_lemmatization=True):
    """
    Perform all NLP preprocessing steps on the input text.
    
    Args:
        text (str): Input text to preprocess.
        remove_stopwords (bool): Whether to remove stopwords.
        use_stemming (bool): Whether to apply stemming.
        use_lemmatization (bool): Whether to apply lemmatization.
    
    Returns:
        str: The preprocessed text.
    """
    # Initialize tools
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # 1. Convert text to lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 5. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 6. Tokenize text
    tokens = word_tokenize(text)
    
    # 7. Remove stopwords (optional)
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    
    # 8. Apply stemming or lemmatization (optional)
    if use_stemming:
        tokens = [stemmer.stem(word) for word in tokens]
    elif use_lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # 9. Rejoin tokens into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Function to truncate the text as specified
def truncate_text(text,n=500):
    words = str(text).split()
    if len(words) > n:
        return ' '.join(words[:n] + words[-1*n:])
    return ' '.join(words)


def inference(url,file_name,save_dir,back_up):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(back_up, exist_ok=True)
    pdf_path = download_pdf_requests(url, save_dir, file_name)
    if pdf_path:
        text =  extract_text_with_backup(file_name,save_dir,back_up)
        if text:
            text = preprocess_text(text)
            text = truncate_text(text,n=300)
        else:
            return None
        model_path = './svm_model.joblib'
        tfidf_path = './tfidf_vectorizer.joblib'
        svm_model = joblib.load(model_path)
        tfidf_vectorizer = joblib.load(tfidf_path)
        X_test_tfidf = tfidf_vectorizer.transform([text])
        y_test_pred_svm = svm_model.predict(X_test_tfidf)
        return y_test_pred_svm[0]
    else:
        return None 