# 📰 Fake News Detection using NLP & Machine Learning

A machine learning classification pipeline built in Python to identify whether news articles are reliable or fabricated based on article metadata (author and title).

---

### 📊 Dataset & Preprocessing
* **Dataset:** Kaggle Fake News Dataset (`20,800` labeled news articles — `0`: Reliable, `1`: Fake).
* **Missing Value Handling:** Replaced null values in `title`, `author`, and `text` fields with empty strings and combined `author` + `title` into a unified `content` feature.
* **NLP Text Cleaning:**
  * Removed non-alphabetic characters using regular expressions (`re`).
  * Converted all text to lowercase and tokenized into words.
  * Removed English stopwords using **NLTK** (`nltk.corpus.stopwords`).
  * Applied **Porter Stemming** (`PorterStemmer`) to reduce words to their root forms.
* **Feature Extraction:** Converted processed text into numerical feature vectors using **TF-IDF Vectorization** (`TfidfVectorizer`).

---

### 🧠 Model & Tech Stack
* **Algorithm:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)
* **Evaluation Metric:** Accuracy Score (`accuracy_score`)
* **Libraries:** `Python`, `NumPy`, `Pandas`, `NLTK`, `Scikit-learn`

---

### 🚀 How to Run
1. Open `Fake_News_Detection.ipynb` in **Google Colab** or **Jupyter Notebook**.
2. Download the Kaggle Fake News dataset (`train.csv`) and update the file path in `pd.read_csv(...)`.
3. Install dependencies if running locally:
   ```bash
   pip install numpy pandas nltk scikit-learn
   ```
4. Run all cells sequentially to preprocess the text, train the Logistic Regression classifier, and evaluate test accuracy.
