IMDB Review Analysis
This project is a sentiment analysis task where the goal is to classify movie reviews from the IMDB dataset into two categories: positive and negative. The notebook utilizes multiple Natural Language Processing (NLP) techniques and machine learning models to preprocess the text, analyze the reviews, and evaluate various vectorization methods and classifiers.

1. Dataset Loading and Initial Exploration
The dataset used for this analysis is the "IMDB Dataset of 50k Movie Reviews" sourced from Kaggle. The data consists of 50,000 movie reviews labeled as either positive or negative. The following steps outline the dataset loading and initial exploration process:

Dataset Download: The dataset is downloaded from Kaggle and extracted into a directory named 'datasets'.
Initial Data Load: The dataset is loaded into a Pandas DataFrame.
Basic Exploration: The basic structure of the data is explored, including checking the unique sentiment labels (positive, negative).
Profiling: A detailed profile of the dataset is generated using the ydata_profiling library to provide insights into the data distribution and missing values.
2. Text Preprocessing and Feature Engineering

In this section, text reviews are cleaned and preprocessed for modeling. The following steps are involved:

Word Count Calculation: A new feature for the number of words in each review is created.
Sentiment Encoding: Sentiment labels (positive/negative) are encoded as binary values (1 for positive, 0 for negative).
Text Cleaning: The reviews are cleaned by removing URLs, HTML tags, and non-alphanumeric characters. All text is converted to lowercase.
Tokenization and Stemming: The reviews are tokenized and stemmed using the Snowball Stemmer from the NLTK library to reduce words to their base forms.
Stopword Removal: Common English stopwords (e.g., "the", "and") are removed to reduce noise in the data.

3. Vectorization Techniques
Various vectorization techniques are evaluated to convert the text data into numerical features that can be used by machine learning models. The following methods are used:

CountVectorizer: Converts the text data into a matrix of token counts.
TfidfVectorizer: Converts the text into a Term Frequency-Inverse Document Frequency (TF-IDF) representation, which reflects the importance of a word in the document relative to the entire corpus.
HashingVectorizer: A variant of CountVectorizer that uses the hash trick to represent text data as a fixed-length feature vector.
These vectorizers are evaluated based on their performance in subsequent models, such as accuracy, training time, and confusion matrix.

4. Model Evaluation
In this section, multiple models are trained and evaluated using the different vectorization techniques. The following steps are carried out:

Logistic Regression: A logistic regression model is trained on the vectorized data to predict the sentiment of the reviews.
Confusion Matrix: For each vectorizer, a confusion matrix is generated to visualize the model’s performance in classifying reviews correctly and incorrectly.
Word2Vec and FastText: Additional models are trained using Word2Vec and FastText embeddings, which are word representation models that capture semantic meaning based on word context. These models are evaluated similarly to the traditional vectorization methods.
Performance Evaluation: Each model's performance is assessed using metrics like accuracy, Mean Squared Error (MSE), and classification reports.

5. Classification with Machine Learning Models
Several classification algorithms are tested to identify the best model for sentiment analysis. The following classifiers are evaluated:

Logistic Regression: A widely used linear model for binary classification.
Multinomial Naive Bayes: A probabilistic classifier based on Bayes' theorem.
Linear Support Vector Classifier (SVC): A linear SVM model for classification tasks.
Random Forest: A powerful ensemble method based on decision trees.
Gradient Boosting: Another ensemble method that builds models sequentially, with each new model correcting the errors of the previous one.
For each classifier, the following steps are conducted:

Pipeline Construction: A pipeline is created using the TfidfVectorizer for feature extraction and the chosen classifier for prediction.
Model Fitting and Prediction: The models are trained and used to predict the sentiment of movie reviews.
Performance Metrics: Each model's accuracy, AUC (Area Under Curve), and confusion matrix are computed. The classification report is generated for detailed performance analysis.

6. Hyperparameter Tuning and Final Evaluation
Once the classifiers are trained, hyperparameter tuning is performed to optimize model performance. The following steps are carried out:

GridSearchCV: Grid search is used to tune hyperparameters for LogisticRegression and RandomForestClassifier. This helps in finding the best combination of hyperparameters for each model.
Best Parameters: The best hyperparameters for each model are identified, and the tuned models are evaluated again.
Model Comparison: A final comparison is made between pre-tuned and post-tuned classifiers based on their accuracy.

7. Conclusion
After evaluating different vectorization techniques and classifiers, the following insights are derived:

Best Performing Models: Logistic Regression and Linear SVC show the highest accuracy, both achieving over 87% accuracy in sentiment classification tasks.
Tuning Impact: Hyperparameter tuning improves the performance of classifiers like Logistic Regression and Random Forest, leading to better results.
Model Recommendation: Based on accuracy, computational efficiency, and interpretability, Logistic Regression is recommended as the best model for this sentiment analysis task.
Future Improvements: The project can be extended by incorporating deep learning-based models or exploring more advanced text preprocessing techniques to improve accuracy further.
Appendix
Dependencies: The following libraries and tools are required to run the notebook: - Pandas. - Numpy. - NLTK. - Scikit-learn - Word2Vec (Gensim) - FastText (Gensim) - Matplotlib, Seaborn (for visualization) - Kaggle API (for downloading the dataset)
File Structure: - datasets/: Directory where the IMDB dataset is stored. - analysis.html: Generated profiling report of the dataset. - results/: Directory where model evaluation results are saved.
