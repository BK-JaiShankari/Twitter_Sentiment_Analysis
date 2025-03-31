# Twitter Sentiment Analysis

This project performs sentiment analysis on Twitter data, focusing on identifying and classifying sentiments (positive, neutral, negative) for specific entities mentioned in the tweets. The dataset used for this project is sourced from Kaggle, and the goal is to classify the sentiment as positive, negative, or neutral for each entity mentioned in the tweets.

## Project Overview

- **Data Source**: Kaggle dataset containing tweets with labeled sentiment for entities.
- **Objective**: To classify sentiment (positive, negative, neutral) for specific entities identified within tweets.
- **Scope**: Implementation of fundamental Natural Language Processing (NLP) techniques, followed by the application of machine learning algorithms to perform sentiment classification on textual data.

## Technologies & Tools

The project utilizes a range of technologies to perform data manipulation, preprocessing, visualization, and model development:

- **Python**: The primary programming language used for data analysis and model development.
- **Pandas**: For data manipulation, cleaning, and handling of large datasets.
- **NumPy**: For numerical operations and efficient array handling.
- **NLTK**: A toolkit for natural language processing tasks, such as tokenization, stop-word removal, and text preprocessing.
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Employed for converting textual data into numerical features that can be used for machine learning models.
- **Scikit-learn**: A comprehensive library for machine learning, utilized for model training, testing, and evaluation.

### Machine Learning Models Implemented:

- **Naïve Bayes**: A probabilistic classifier based on Bayes' Theorem, ideal for text classification tasks.
- **Logistic Regression**: A linear model for binary classification that has been extended to multi-class classification for sentiment analysis.
- **Support Vector Machine (SVM)**: A supervised learning model used for classification and regression tasks, chosen for its efficiency in high-dimensional spaces.
- **Random Forest**: An ensemble learning method utilizing multiple decision trees to improve classification performance.

### EDA (Exploratory Data Analysis) & Visualization:
- **Word Cloud**: To visualize the most frequent words used in the tweets.
- **Distribution per Entity**: Analyzing and visualizing sentiment distribution across different entities.

### Performance Evaluation:
Model performance is evaluated using metrics such as accuracy, precision, recall, and confusion matrix. Continuous refinement of the models is performed to improve predictive accuracy.

## Objective

The primary objectives of this project are to:

- **Data Preprocessing**: Clean and preprocess the tweet data, including tokenization, stopword removal, and the conversion of text to numerical format using TF-IDF.
- **Model Development**: Train various machine learning models (Naïve Bayes, Logistic Regression, SVM, and Random Forest) to predict sentiment for different entities mentioned in tweets.
- **Model Evaluation and Improvement**: Evaluate model performance using metrics such as accuracy, precision, recall, and confusion matrix. Continuously refine the models to improve predictive accuracy.
