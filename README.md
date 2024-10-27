# Predicting Amazon Movie Review Ratings: A Machine Learning Approach

## Introduction

The objective of this project is to predict star ratings associated with user reviews from Amazon Movie Reviews using the available features provided in `train.csv`. The challenge involves developing a model that accurately predicts the `Score` (rating from 1 to 5) based on textual reviews and other metadata. This report outlines the data exploration, feature engineering, model selection, and optimization processes undertaken to achieve optimal predictive performance.

## Data Exploration and Preprocessing

### Handling Missing Values

- **Missing `Score` Values**: Identified 212,192 rows with missing `Score` values in `train.csv`. These rows were removed from the training set as they represent the target variable we aim to predict.
- **Missing `Text` and `Summary` Values**: Found 54 missing `Text` entries and 32 missing `Summary` entries after removing rows with missing `Score`. Rows with missing `Text` were dropped due to the importance of review text in prediction. Missing `Summary` values were filled with empty strings.

### Data Types and Conversion

- **Converting `Score` to Integer**: The `Score` column was converted from float to integer for classification purposes.

- **Converting `Time` to Datetime**: The `Time` column, originally a UNIX timestamp, was converted to a datetime object, enabling extraction of temporal features.

## Feature Engineering

### Textual Features

- **Combining `Summary` and `Text`**: Merged these two columns into a single `ReviewText` field to capture all textual information provided by the user.

- **Text Preprocessing**:

  - **Lowercasing and Removing Non-Alphabetic Characters**: Standardized the text by converting to lowercase and removing numbers and special characters.

  - **Tokenization and Stop Word Removal**: Split the text into tokens and removed common English stop words using NLTK's stop word list.

  - **Stemming**: Applied Snowball Stemmer to reduce words to their root forms, thereby normalizing variations of the same word.

- **TF-IDF Vectorization**:

  - Experimented with different values of `max_features` in `TfidfVectorizer`. Found that setting `max_features=11,000` provided the best validation accuracy.

  - Transformed the preprocessed text into TF-IDF features to represent the importance of words in the corpus.

### Numerical Features

- **Helpfulness Ratio**: Calculated as `HelpfulnessNumerator / HelpfulnessDenominator`. Replaced zero denominators with NaN to avoid division by zero, then filled resulting NaN values with 0.0, assuming no helpfulness feedback.

- **Review Age in Days**: Computed the age of each review by subtracting the review date from the current date.

- **Year of Review**: Extracted the year from the `ReviewTime` datetime object.

- **Feature Selection**:

  - Experimented with different combinations of numerical features.

  - Found that using only `HelpfulnessRatio` and `ReviewYear` as numerical features yielded the highest validation accuracy.

## Handling Class Imbalance

- Observed significant class imbalance with a majority of reviews having a `Score` of 5:

  Score Distribution: 5: 793,163, 4: 335,228, 3: 176,082, 2: 89,678, 1: 91,190

- **Solution**: Applied `class_weight='balanced'` in the classifier to adjust weights inversely proportional to class frequencies, improving the model's ability to learn from minority classes.

## Model Selection and Training

### Classifier Choice

- **Logistic Regression**: Initial experiments with `LogisticRegression` were computationally intensive due to dataset size.

- **Random Forest**: Tried `RandomForestClassifier` but faced long training times and overfitting.

- **Stochastic Gradient Descent Classifier**: Selected `SGDClassifier` with `loss='log_loss'` for its efficiency on large datasets and support for multiclass classification. Found that `SGDClassifier` provided the best trade-off between training time and performance.

### Hyperparameter Tuning

- **Loss Function and Penalty**: Experimented with different loss functions (`hinge`, `log_loss`) and penalties (`l1`, `l2`). Discovered that `loss='log_loss'` and `penalty='l2'` achieved optimal results.

- **Early Stopping**: Implemented `early_stopping=True` to prevent overfitting by monitoring the validation loss and halting training when improvement ceased.

### Model Training

- **Training-Validation Split**: Split the data into training and validation sets with a 75:25 ratio, ensuring stratification to maintain class distribution.

- **Model Fitting**: Trained the `SGDClassifier` on the training data, combining TF-IDF features and selected numerical features.

- **Performance Evaluation**: Achieved a validation accuracy of **63.04%**. The classification report indicated improved precision and recall across all classes compared to models without class weighting.

## Attempts at Data Augmentation

- **Synonym Replacement**: Implemented data augmentation through synonym replacement using WordNet to increase dataset diversity. Augmented 10% of the data by replacing words with their synonyms.

- **Outcome**: The augmented model showed a slight decrease in validation accuracy to **62.71%**. Decided not to include data augmentation in the final model due to lack of performance improvement.

## Final Model and Results

- **Features Used**:

  - **Textual Features**: TF-IDF vectors of preprocessed `ReviewText` with `max_features=11,000`.
  - **Numerical Features**: `HelpfulnessRatio` and `ReviewYear` scaled using `StandardScaler`.

- **Classifier**: `SGDClassifier` with `loss='log_loss'`, `penalty='l2'`, `class_weight='balanced'`, and `early_stopping=True`.

- **Validation Performance**:

  - **Validation Accuracy**: 63.04%
  - **Classification Report**:

    | Score | Precision | Recall | F1-score | Support |
    | ----- | --------- | ------ | -------- | ------- |
    | 1     | 0.51      | 0.63   | 0.56     | 22,797  |
    | 2     | 0.36      | 0.36   | 0.36     | 22,418  |
    | 3     | 0.43      | 0.35   | 0.39     | 44,019  |
    | 4     | 0.47      | 0.35   | 0.40     | 83,804  |
    | 5     | 0.75      | 0.84   | 0.79     | 198,284 |

    - Overall Accuracy: 63.04%
    - Macro Average: Precision = 0.50, Recall = 0.51, F1-score = 0.50
    - Weighted Average: Precision = 0.61, Recall = 0.63, F1-score = 0.62

- **Observations**: The model performs best on the majority class (`Score` = 5), as expected due to class imbalance. Incorporating `class_weight='balanced'` significantly improved recall for minority classes.

## Conclusion

Through extensive experimentation with feature selection, classifier tuning, and handling class imbalance, the final model achieved a validation accuracy of **63.04%**. Key strategies that contributed to performance improvement include:

- **Selective Feature Inclusion**: Focusing on `HelpfulnessRatio` and `ReviewYear` as numerical features provided better results than including all available numerical features.

- **Adjusting `max_features` in TF-IDF**: Optimizing the number of features to 11,000 balanced the trade-off between capturing meaningful vocabulary and managing computational resources.

- **Handling Class Imbalance**: Using `class_weight='balanced'` in the classifier improved the model's ability to predict minority classes.

- **Preventing Overfitting**: Implementing early stopping in the `SGDClassifier` prevented the model from overfitting to the training data.

The exploration process highlighted the importance of feature selection and model tuning in handling large, imbalanced datasets. While data augmentation did not yield positive results in this case, it remains a valuable technique worth exploring further with different approaches.

---

**References**:

- Scikit-learn documentation: [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- NLTK documentation: [Stemming and Stopwords](https://www.nltk.org/)
- Kaggle discussions and forums for insights on handling imbalanced datasets and text preprocessing techniques.
