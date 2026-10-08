# SMS Spam Detection using NLP

This project implements an **SMS Spam Detection system** using Natural Language Processing (NLP) and machine learning.

The project preprocesses SMS messages, converts text into numerical features using **CountVectorizer** and **TF-IDF**, and compares multiple machine-learning classification algorithms.

## Project Workflow

```text
SMS Dataset
     ↓
Data Cleaning
     ↓
Text Preprocessing
     ↓
CountVectorizer / TF-IDF
     ↓
Machine Learning Models
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
SMS Prediction
```

## Dataset

The project uses an SMS spam classification dataset containing messages labelled as either:

* `ham` → Normal message
* `spam` → Spam message

The notebook expects the dataset file:

```text
spam.csv
```

The original dataset columns are renamed to:

```text
label
message
```

The dataset initially contains **5,572 records**.

After removing **403 duplicate records**, the dataset contains **5,169 records**.

The resulting class distribution is:

* Ham: 4,516
* Spam: 653

## Text Preprocessing

The SMS messages are cleaned before feature extraction.

The preprocessing includes:

* Converting text to lowercase
* Removing URLs
* Removing punctuation
* Removing numbers
* Removing extra whitespace

The processed text is stored in:

```text
clean_message
```

## Exploratory Data Analysis

The notebook performs:

* Dataset inspection
* Missing-value checking
* Duplicate detection
* Spam/Ham class distribution
* Word clouds
* Confusion matrices
* Classification reports

## Feature Extraction

Two different text-vectorization techniques are used.

### CountVectorizer

CountVectorizer converts the messages into numerical vectors based on word frequency.

```python
CountVectorizer(stop_words="english")
```

### TF-IDF

TF-IDF assigns importance to words based on how frequently they appear in a document compared with the complete dataset.

```python
TfidfVectorizer(stop_words="english")
```

## Train-Test Split

The dataset is divided into:

* 80% Training data
* 20% Testing data

The split uses:

```python
random_state=42
```

and stratification is used to maintain the spam/ham distribution.

## Machine Learning Models

The following models are compared:

* Logistic Regression
* Multinomial Naive Bayes
* Linear SVM
* Decision Tree
* Random Forest

## CountVectorizer Results

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   97.39% |   100.00% | 79.39% |   88.51% |
| Naive Bayes         |   97.87% |    93.60% | 89.31% |   91.41% |
| Linear SVM          |   97.87% |    99.10% | 83.97% |   90.91% |
| Decision Tree       |   95.74% |    87.18% | 77.86% |   82.26% |
| Random Forest       |   97.20% |    98.11% | 79.39% |   87.76% |

## TF-IDF Results

| Model               |   Accuracy |  Precision | Recall |   F1-Score |
| ------------------- | ---------: | ---------: | -----: | ---------: |
| Linear SVM          | **98.07%** | **99.12%** | 85.50% | **91.80%** |
| Random Forest       |     97.00% |     99.02% | 77.10% |     86.70% |
| Naive Bayes         |     96.23% |     98.94% | 70.99% |     82.67% |
| Logistic Regression |     94.87% |     98.75% | 60.31% |     74.88% |
| Decision Tree       |     94.78% |     81.30% | 76.34% |     78.74% |

## Best Model

The best-performing model is:

**TF-IDF + Linear SVM**

Performance:

* Accuracy: **98.07%**
* Precision: **99.12%**
* Recall: **85.50%**
* F1-Score: **91.80%**

The trained Linear SVM model can be used to classify new SMS messages as either **SPAM** or **HAM**.

## SMS Prediction

The notebook contains a prediction function:

```python
predict_sms(message)
```

The function:

1. Cleans the input SMS
2. Converts it into TF-IDF features
3. Passes the features to the trained model
4. Returns the predicted class

Example output:

```text
SPAM
```

or

```text
HAM
```

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* WordCloud
* Jupyter Notebook

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd SMS-Spam-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook and run the cells.

## Project Structure

```text
SMS-Spam-Detection/
│
├── ml(3).ipynb
├── spam.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## Key Learning Outcomes

This project demonstrates:

* Natural Language Processing
* Text preprocessing
* Exploratory Data Analysis
* CountVectorizer
* TF-IDF
* Train-test splitting
* Stratified sampling
* Logistic Regression
* Naive Bayes
* Linear SVM
* Decision Trees
* Random Forest
* Confusion matrices
* Precision, Recall and F1-Score
* Model comparison
* Text classification

## Disclaimer

This project is created for **educational and machine-learning purposes**.

## Author

**Palak Bajaj**

B.Tech CSE | VIT-AP University
