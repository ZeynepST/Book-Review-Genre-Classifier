# **Book Review Genre Classifier**

This project is a machine learning model that classifies book reviews into specific genres. It uses natural language processing (NLP) techniques
to analyze review text and assign a genre based on keywords and content. The project follows a full NLP pipeline, from data preprocessing and feature engineering to model training and evaluation.

# **🚀Project Plan and Methodology🚀**

The core of this project is to use the Review column as the feature and create a new Genre column as the label. The classification is performed by:

**Genre Assignment:** A dictionary of common words is created for seven genres: romance, sci-fi, historical, fantasy, children, thriller, and adventure. Each review is then assigned a genre based on which set of keywords it contains the most.

**Text Preprocessing:** The Review text is preprocessed by tokenizing, converting to lowercase, and removing punctuation and stop words using the gensim.utils.simple_preprocess function.

**Feature Engineering:** The preprocessed text is transformed into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.

**Model Training and Evaluation:** Four different machine learning models are trained and evaluated using a grid search to find the best hyperparameters:

  - Logistic Regression

  - Random Forest Classifier

  - Decision Tree Classifier

  - Gradient Boosting Classifier (GBDT)

**Performance Metrics:** Model performance is evaluated using both accuracy and balanced accuracy scores, the latter being particularly useful for imbalanced datasets.

# **🧩Technologies and Libraries🧩**
**Pandas:** Used for data manipulation, including loading the dataset and creating new columns.

**NumPy:** Used for numerical operations.

**Matplotlib and Seaborn:** Used for data visualization and exploration.

**scikit-learn:** A comprehensive library for machine learning, used for model training (LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier), model selection (GridSearchCV), and performance metrics (accuracy_score, balanced_accuracy_score, confusion_matrix).

**Gensim:** Used for text preprocessing and tokenization.

# **📓Dataset📓**
BookReviewsDataSet.csv: The dataset used in the project, which contains book reviews.

How to Run the Project
Ensure you have Python installed.

Install the required libraries. The notebook uses pandas, numpy, matplotlib, seaborn, scikit-learn, gensim, and nltk. You can install them with pip:

```
  Bash
  pip install pandas numpy matplotlib seaborn scikit-learn gensim nltk
```
Place the BookReviewsDataSet.csv file in a data directory within the project's root folder.
Open and run the BookReviewGenreClassifier.ipynb Jupyter Notebook. The notebook contains all the code cells for data loading, preprocessing, model training, and evaluation.






