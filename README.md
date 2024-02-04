# Resume Classification Project using Naive Bayes Classifier

## Overview

This repository documents a machine learning project for the classification of resumes into different categories using the UpdatedResumeDataset.csv dataset. The project follows a systematic approach, including pre-processing, exploratory data analysis (EDA), text cleaning, stopwords removal, feature extraction, and model evaluation.

## Steps

### 1. Read the UpdatedResumeDataset.csv dataset

- Loaded the dataset using the pandas library to initiate the project.

### 2. Displayed Categories and Counts
   
- Examined the distribution of resume categories within the dataset using the `value_counts()` method. This step provides an initial understanding of the dataset's class distribution.

### 3. Created a Count Plot
   
- Visualized the count of resumes for each category using a horizontal bar plot. This plot provides a clear representation of the number of resumes in each category, aiding in identifying any class imbalances.

### 4. Created a Pie Plot

- Generated a pie chart illustrating the percentage distribution of resumes across different categories. This visualization helps in grasping the proportional contribution of each category to the overall dataset.

### 5. Converted all the Resume text to lower case

- Standardized the text data by converting all resume text to lowercase.

### 6. Defined a function to clean the resume text

- Developed a cleaning function to remove special characters, URLs, RT, punctuations, and extra whitespace.
- Stored the cleaned text in a new column for further analysis.

### 7. Used nltk package to find the most common words and Generate Word Cloud

- Utilized the nltk library to tokenize the cleaned resume text and identify the most common words.
- Created a Word Cloud to visually represent the most frequently occurring words in the resume text.

### 8. Converted the categorical variable Category to a numerical feature

- Encoded the categorical variable 'Category' into numerical values using label encoding.

### 9. Converted Text to Feature Vectors (TF-IDF)
   
- Utilized the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert the cleaned resume text into numerical feature vectors. This process involves tokenizing documents, learning vocabulary, and calculating inverse document frequency weightings.

### 10. Applied Naive Bayes Classifier
    
- Splitted the data into training and testing sets.
- Implemented a Naive Bayes Classifier, specifically the MultinomialNB model, to train on the feature vectors and make predictions.
- Evaluated the model's performance by calculating accuracy and providing a detailed classification report.

## Project Link

Explore the complete project, including the Jupyter notebook with code and visualizations, at [Project Link](https://colab.research.google.com/drive/1-Y7YM9YLctt2BdncASrJx9Vdr8-B22dT?usp=sharing).

Feel free to adapt and use the provided code for your own resume classification tasks. For any questions or suggestions, please open an issue or reach out. Happy exploring!
