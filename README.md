# SVM_long_text_prediction
code and inference can be seen in [svm_classifier.ipynb](https://github.com/BhaveshRaj27/SVM_long_text_prediction/blob/main/svm_classifier.ipynb) and inference function can be seen in [utils.py](https://github.com/BhaveshRaj27/SVM_long_text_prediction/blob/main/utils.py) in detail. App creation [file](https://github.com/BhaveshRaj27/SVM_long_text_prediction/blob/main/app.py)

[Link for app](https://svm-long-text-prediction-bhavesh.streamlit.app/) 

## How long did it take to solve the problem?
 - It took me around 10 hours to solve the problem which also includes reading some research papers.


## Explain your solution? 
## Approach Summary
- In the given classification problem, I am downloading pdf using requests and wget libraries and extracting text from pdf using PyPDFLoader from langchain.document_loader. after extraction using text truncation (head+tail) to overcome the long text problem. Furthe used the TF-IDF method for vectorization, SVM for modelling and f1 score to create metrics.

## Approach summary in detail.
![image (11)](https://github.com/user-attachments/assets/20b4e199-f230-43c7-9a3a-ec12ac27291d)

The figure above provides the overall workflow for the given problem.


- **Extracting pdfs links from excel** - using -pandas to read the excel.
- **Downloading pdf from the links in excel** -  using requests and wget  library to download the pdfs as different  
- **Extracting text from downloaded pdf**. Extracting text from pdf using Langchain pdf reader
- **Cleaning extracted text**: applying below 8 cleaning and text processing steps for better result
  Lowercase
  Urls removing
  HTML tag removal
  Number removal
  Punctuation removal
  Tokenization
  Stop word removal
  Lemmatization
- **EDA on cleaned text** - checking text length, checking data imbalance etc
- 
  ## Add image  
- **Truncating text** -  This technique is used to tackle long text in classification problems. We take n number of words from the start and end of the document (head+ tail). There are multiple approaches such as head only, tail only, hier. mean, hier. max, hier. self-attention etc but head+ tail truncation performed best. see the [research paper](https://arxiv.org/abs/1905.05583) 
 
Train, validation and test splitting: splitting data into train, validation and test split which helps to analyze the model for overfitting and under-fitting cases.
![image (15)](https://github.com/user-attachments/assets/5c7c54df-cebb-4763-834d-f401cf586252)

- **Vectorization** - Using TF-IDF to convert text into vector format. It is a fast approach, with low latency time, which is helpful during inference and scalability. Further, it overcomes the problem of BoW by penalizing common words (low IDF), it prevents them from dominating the representation. It also adjusts term frequencies by dividing them by the total number of terms, ensuring fair comparison across documents of different lengths.

  ### Which model did you use and why?

- **Model selection SVM:** Selecting SVM, the dataset has a small number of data points around ~2000, also SVM is good at handling large dimensionality vectors such as TF-IDF. Easy to finetune and handle non-linear problems such as current task using kernels like RBF.

 ### Report the model's performance on the test data using an appropriate
metric. Explain why you chose this particular metric.
- **Metric selection (F1 score):** Selecting weighted F1 score as metric, as it penalizes both False positive and false negative if precision or recall is low and focuses on minority class. Maintain the balance between precision and recall. In the end, it is a  good metric for multi-class scenarios. F1 score can be calculated for each class and averaged (macro or weighted) to evaluate performance in multi-class classification. Also I have used the Confusion matrix to see at which class our model is not performing well.
- **Hyperparameter optimization for the length of Truncating Text and SVM:** Optimizing length of truncation, vector length,  C and gamma to reduce overfitting adding plots of optimization. Optimizing truncation length  and other parameters in figure:
- ![image (12)](https://github.com/user-attachments/assets/ad916cde-162d-4626-9e1b-83f45eeee1b0)
![image (13)](https://github.com/user-attachments/assets/3f07481a-a4f0-4ee1-820b-4cc77a53bcd6)

- **Inferencing:** App creation - Creating App, using streamlit.

## Result
f1_score_train :  0.9938944038673142

f1_score_validation :  0.9848576219300204

f1_score_test :  0.963480668922752

The f1 score for the test dataset is 0.96.

![image (14)](https://github.com/user-attachments/assets/4bdf9e55-28cc-4df3-b56c-31eb022f0f85)

### Any shortcomings and how can we improve the performance?
- With the f1 score shown model performs good on training and validation data but still f1 score for test data shows a significant difference with training which suggests that the model might be still overfitted. As I have already optimized the hyperparameters some more data (as many of the links are not working and some pdf are corrupted) can be added to the training data set. Or we can use a lengthy approach like data augmentation (synonym replacement, or back-translation) which can help us to increase the training dataset but require significant time.
- Secondly, I can use techniques like Summarization and chunking(then using voting)) which helps us to better understand the text. But Summarization is time-consuming and resource intensive, while chunking may create an imbalance dataset problem as in our dataset length of text went from 400 to 60,000.
- Using more advanced vectorization techniques such as Word2vec or Glove which help to understand the semantic meaning of text and can provide better result.
- Using models like BERT, they need more data to generalise but are time and  resource-intensive. 


