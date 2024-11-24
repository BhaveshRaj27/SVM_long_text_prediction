# SVM_long_text_prediction
## Approach Summary
- In the given classfication problem, I am downloading pdf using requsts and wget librarues and extracting text from pdf using PyPDFLoader from langchain.document_loader. after extraction using text truncation (head+tail) to overcome the long text probelm. Furthe used TF-IDF method for vectorization, SVM for modeling and f1 score to create metrics.

## Approch summary in detail.
##Add image
The figue above provide the overall work flow for the given probelm.


- **Extracting pdfs links from excel** - using -pandas to read the excel.
- **Downloading pdf from the links in excel** -  using requests and wget  library to download the pdfs as different  
- **Extracting text from downloaded pdf**. Extracting text from pdf using Langchain pdf reader
Cleaning extracted text : applying below 8 cleaning and text processing step for better result
Lowercase
Urls removing
HTML tag removal
Number removal
Punctuation removal
Tokenization
Stop word removal
Lemmatization
EDA on cleaned text - checking text length, checking data imbalance etc  
Truncating text -  Technique used to tackle long text in classification problems. We take n number of words from the start and end of the document (head+ tail). 
Train, validation and test splitting: splitting data into train, validation and test split which help to analyze model for overfitting and under-fitting cases.
Vectorization - Using TF-IDF to convert text in vector format. It is a fast approach, with low latency time, which is helpful at the time of inference and scalability. Further, it overcome the problem of BoW by penalizing common words (low IDF), it prevents them from dominating the representation.It also adjusts term frequencies by dividing them by the total number of terms, ensuring fair comparison across documents of different lengths.

Model selection SVM : Selecting SVM, the data set has a small number of data points around ~2000, SVM is good in handling large dimensionality vectors such TF-IDF.  Robust to overfitting and handling non linear problems using kernels like rbf.
Metric selection (F1 score): Selecting weighted F1 score as metric, penalize if  precision or recall is low and focus on minority class.
Hyperparameter optimization for length of Truncating Text and SVM: Optimizing length of truncation, vector length,  C and gamma to reduce overfitting
Inferencing: App creation - Creating App, using streamlit.
