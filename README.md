# Predicting Amazon Movie Review Ratings: A Machine Learning Approach

## Introduction

The objective of this project is to predict star ratings associated with user reviews from Amazon Movie Reviews using the available features provided in `train.csv`. The challenge involves developing a model that accurately predicts the `Score` (rating from 1 to 5) based on textual reviews and other metadata. This report outlines the data exploration, feature engineering, model selection, and optimization processes undertaken to achieve optimal predictive performance.

## Data Exploration and Preprocessing

### Handling Missing Values

- **Missing `Score` Values**: Identified 212,192 rows with missing `Score` values in `train.csv`. These rows were removed from the training set as they represent the target variable we aim to predict.

- **Missing `Text` and `Summary` Values**: Found 54 missing `Text` entries and 32 missing `Summary` entries after removing rows with missing `Score`. Rows with missing `Text` were dropped due to the importance of review text in prediction. Missing `Summary` values were filled with empty strings to maintain consistency in text processing.

### Data Types and Conversion

- **Converting `Score` to Integer**: The `Score` column was converted from float to integer for classification purposes.

- **Converting `Time` to Datetime**: The `Time` column, originally a UNIX timestamp, was converted to a datetime object, enabling extraction of temporal features such as review year.

## Feature Engineering

### Textual Features

- **Combining `Summary` and `Text`**: Merged these two columns into a single `ReviewText` field to capture all textual information provided by the user.

- **Text Preprocessing**:

  - **Lowercasing and Removing Non-Alphabetic Characters**: Standardized the text by converting to lowercase and removing numbers and special characters using regular expressions.
  - **Tokenization and Stop Word Removal**: Split the text into tokens and removed common English stop words using NLTK's stop word list.
  - **Stemming**: Applied Snowball Stemmer from NLTK to reduce words to their root forms, thereby normalizing variations of the same word (e.g., "running" to "run").

- **TF-IDF Vectorization**:
  - Experimented with different values of `max_features` in `TfidfVectorizer`. Found that setting `max_features=11,000` provided the best validation accuracy, balancing the vocabulary size and computational efficiency.
  - Transformed the preprocessed text into TF-IDF features to represent the importance of words in the corpus.

### Numerical Features

- **Helpfulness Ratio**: Calculated as `HelpfulnessNumerator / HelpfulnessDenominator`. Replaced zero denominators with `NaN` to avoid division by zero, then filled resulting `NaN` values with 0.0, assuming no helpfulness feedback was provided.

- **Review Year**: Extracted the year from the `ReviewTime` datetime object to capture temporal trends in reviews.

- **Feature Selection**:
  - Experimented with different combinations of numerical features, including `ReviewLength`, `ReviewMonth`, `ReviewDayOfWeek`, and `ReviewAgeDays`.
  - Found that using only `HelpfulnessRatio` and `ReviewYear` as numerical features yielded the highest validation accuracy.

## Handling Class Imbalance

- Observed significant class imbalance with a majority of reviews having a `Score` of 5:

  - **Score Distribution**: 5: 793,163; 4: 335,228; 3: 176,082; 2: 89,678; 1: 91,190.

- **Solution**: Applied `class_weight='balanced'` in the classifier to adjust weights inversely proportional to class frequencies, improving the model's ability to learn from minority classes.

## Model Selection and Training

### Classifier Choice

- **Logistic Regression**: Initial experiments with `LogisticRegression` were computationally intensive and time-consuming due to dataset size.

- **Random Forest**: Attempted using `RandomForestClassifier`, but it resulted in long training times and overfitting on the training data.

- **Stochastic Gradient Descent Classifier**: Selected `SGDClassifier` with `loss='log_loss'` (logistic regression) for its efficiency on large datasets and support for multiclass classification. The `SGDClassifier` provided the best trade-off between training time and performance, completing training in a reasonable time frame.

### Hyperparameter Tuning

- **Loss Function and Penalty**: Experimented with different loss functions (`hinge`, `log_loss`) and penalties (`l1`, `l2`, `elasticnet`). Discovered that `loss='log_loss'` and `penalty='l2'` achieved optimal results, providing better convergence and performance.

- **Early Stopping**: Implemented `early_stopping=True` with `validation_fraction=0.1` and `n_iter_no_change=5` to prevent overfitting by monitoring the validation loss and halting training when improvement ceased.

- **Learning Rate and Max Iterations**: Set `max_iter=500` to allow sufficient epochs for convergence. Used default learning rate with adaptive adjustments inherent in `SGDClassifier`.

### Model Training

- **Training-Validation Split**: Split the data into training and validation sets with a 75:25 ratio using `train_test_split`, ensuring stratification to maintain class distribution across splits (`stratify=y`). Set `random_state=42` for reproducibility of results.

- **Model Fitting**: Trained the `SGDClassifier` on the training data, combining TF-IDF features and selected numerical features. Used `n_jobs=-1` to utilize all available CPU cores for faster computation.

### Performance Evaluation

- **Validation Accuracy**: Achieved a validation accuracy of **63.04%**.

- **Classification Report**:

  - The classification report indicated improved precision and recall across all classes compared to models without class weighting.

- **Validation Results**:

  | Score | Precision | Recall | F1-Score | Support |
  | ----- | --------- | ------ | -------- | ------- |
  | 1     | 0.51      | 0.63   | 0.56     | 22,797  |
  | 2     | 0.36      | 0.36   | 0.36     | 22,418  |
  | 3     | 0.43      | 0.35   | 0.39     | 44,019  |
  | 4     | 0.47      | 0.35   | 0.40     | 83,804  |
  | 5     | 0.75      | 0.84   | 0.79     | 198,284 |

  **Accuracy**: 0.63 (371,322 samples)

  **Macro Avg**: Precision 0.50, Recall 0.51, F1-Score 0.50

  **Weighted Avg**: Precision 0.61, Recall 0.63, F1-Score 0.62

## Attempts at Data Augmentation

- **Synonym Replacement**: Implemented data augmentation through synonym replacement using NLTK's WordNet to increase dataset diversity. Augmented 10% of the data by randomly replacing words with their synonyms in the `ReviewText`.

- **Outcome**: The augmented model showed a slight decrease in validation accuracy to **62.71%**. Possible reasons for the decrease include:
  - Synonym replacement may have introduced noise, altering the original sentiment or meaning of reviews.
  - The model may have struggled to generalize with augmented data that does not accurately reflect real-world variations.
  - Decided not to include data augmentation in the final model due to lack of performance improvement.

## Final Model and Results

- **Features Used**:

  - **Textual Features**: TF-IDF vectors of preprocessed `ReviewText` with `max_features=11,000`.
  - **Numerical Features**: `HelpfulnessRatio` and `ReviewYear` scaled using `StandardScaler`.

- **Classifier**: `SGDClassifier` with:

  - `loss='log_loss'` (logistic regression)
  - `penalty='l2'`
  - `class_weight='balanced'`
  - `early_stopping=True`
  - `max_iter=500`
  - `random_state=42` for reproducibility

- **Validation Performance**: Achieved a validation accuracy of **63.04%** with improved metrics for minority classes.

- **Observations**: The model performs best on the majority class (`Score` = 5), as expected due to class imbalance. Incorporating `class_weight='balanced'` significantly improved recall for minority classes, especially for `Score` = 1.

- **Submission Generation**: Applied the same preprocessing and feature extraction steps to the test data. Ensured consistent handling of missing values and data types. Predicted `Score` for the test data using the trained model and created the submission file `submission.csv` in the required format.

## Conclusion

Through extensive experimentation with feature selection, classifier tuning, and handling class imbalance, the final model achieved a validation accuracy of **63.04%**. Key strategies that contributed to performance improvement include:

- **Selective Feature Inclusion**: Focusing on `HelpfulnessRatio` and `ReviewYear` as numerical features provided better results than including all available numerical features, likely due to reducing noise and overfitting.

- **Adjusting `max_features` in TF-IDF**: Optimizing the number of features to 11,000 balanced the trade-off between capturing meaningful vocabulary and managing computational resources.

- **Handling Class Imbalance**: Using `class_weight='balanced'` in the classifier improved the model's ability to predict minority classes, enhancing overall performance.

- **Preventing Overfitting**: Implementing early stopping in the `SGDClassifier` prevented the model from overfitting to the training data, as evidenced by stable validation accuracy.

- **Efficient Model Selection**: Choosing `SGDClassifier` allowed efficient training on a large dataset without sacrificing accuracy.

The exploration process highlighted the importance of feature selection and model tuning in handling large, imbalanced datasets. While data augmentation did not yield positive results in this case, it remains a valuable technique worth exploring further with different approaches or datasets.

## External Resources and References

- Scikit-learn documentation: [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- NLTK documentation: [Stemming and Stopwords](https://www.nltk.org/)
- [CSDN](https://so.csdn.net/so/search?spm=1001.2015.3001.4498&q=%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7&t=&u=) forums for insights on handling imbalanced datasets and text preprocessing techniques.
- Data resource: [Kaggle](https://www.kaggle.com/competitions/cs-506-midterm-fall-2024/)
