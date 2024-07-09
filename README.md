# Project Overview

This project consists of two Jupyter notebooks, `Main.ipynb` and `Extra.ipynb`, which are used for text classification and natural language processing tasks. Below is an overview and detailed description of the contents and functionalities provided by each notebook.

## Main.ipynb

The `Main.ipynb` notebook focuses on text preprocessing, feature extraction using TF-IDF, and text classification using the Naive Bayes algorithm. It includes the following sections:

### Sections

1. **Data Loading and Exploration**
   - Load dataset.
   - Explore dataset with visualizations like histograms to understand text length distributions across genres.
   
2. **Word Frequency Analysis**
   - Analyze word frequency in text descriptions.
   
3. **Data Cleaning**
   - Clean the text data by removing stop words, punctuation, and performing other preprocessing tasks.
   
4. **TF-IDF**
   - Calculate Term Frequency-Inverse Document Frequency (TF-IDF) for the dataset.
   
5. **Test-Train Split**
   - Split the data into training and testing sets.
   - Visualize the genre distribution in train and test sets.
   
6. **Modeling**
   - Implement and train a Naive Bayes classifier.
   - Test different alpha values to find the best performing model based on validation scores.
   - Save and reload the model for future use.
   
7. **Evaluation**
   - Clean the test data and apply the trained TF-IDF model.
   - Evaluate the model's performance on the test set.

## Extra.ipynb

The `Extra.ipynb` notebook complements the main notebook by focusing on more advanced techniques and additional processing steps. It includes the following sections:

### Sections

1. **Library Installation**
   - Instructions for installing necessary libraries: `transformers`, `torch`, and `scikit-learn`.

2. **Data Preparation**
   - Load libraries like `pandas` and `train_test_split` from `scikit-learn`.
   - Read data from CSV files and split it into training and validation sets.
   
3. **Tokenization**
   - Load the `DistilBertTokenizer` from `transformers`.
   - Define a function `encode_data` for tokenizing texts.
   - Tokenize the training, validation, and test data.
   
4. **Dataset Creation**
   - Load libraries like `torch` and `DataLoader`.
   - Create a `MovieDataset` class to manage tokenized data and labels.
   - Create datasets for training, validation, and testing.
   
5. **Model Training**
   - Load `DistilBertForSequenceClassification`, `AdamW`, and `get_linear_schedule_with_warmup` from `transformers`.
   - Load the DistilBERT model for sequence classification.
   - Move the model to CUDA if available.
   - Create `DataLoader` for training and validation sets.
   - Define optimizer and learning rate scheduler.
   - Train the model.
   
6. **Evaluation**
   - Evaluate the trained model on the validation and test sets.
   
## How to Use

1. **Install Required Libraries**
   Make sure to install the required libraries before running the notebooks:
   ```bash
   pip install transformers torch scikit-learn
