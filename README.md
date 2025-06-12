# Iris Classifier

---

## ℹ️ About the Project

**Iris Classifier** is project that analyzes:

- Analyze Iris Species pattern based on Sepal Features
- Analyze Iris Species pattern based on Petal Features
- Analyze Iris Features Distribution
- Analyze Iris Features Correlation
- Classify Features based on Iris Species and evaluates Accuracy and Standard Deviation, using:
  - Linear SVC (Support Vector Classification)
  - K-Neighbors Classifier
  - SVC
  - Random Forest
  - Gradient Boosting
  - Voting
  - Logistic Regression
  - Decision Tree

---

## 🛠️ Built With

- [Python](https://www.python.org/) — primary programming language
- [PyCharm](https://www.jetbrains.com/pycharm/) — IDE

- [Scikit-learn](https://scikit-learn.org/stable/) — machine learning
- [Pandas](https://pandas.pydata.org/) — data manipulation
- [NumPy](https://numpy.org/) — number operations
- [Matplotlib](https://matplotlib.org/) — plotting
- [Seaborn](https://seaborn.pydata.org/) — data visualization

---

## 📦 Getting Started

### Prerequisites

To run the project locally, you'll need:

- PyCharm (2025.1.1.1 or newer)

---

### Installation & Setup

1. **Install the necessary libraries:**

   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn kaggle

2. **Download dataset through the terminal:**

   ```bash
   kaggle datasets download -d uciml/iris

3. **Extract the dataset zip:**

   ```bash
   with zipfile.ZipFile("iris.zip", "r") as zip_ref:
      zip_ref.extractall("iris")

4. **Run the code:**

   Run the provided `code.py`

---

## 📊 Results

![Alt text](Figure_1.png?raw=true "Title")
![Alt text](Figure_2.png?raw=true "Title")
![Alt text](Figure_3.png?raw=true "Title")
![Alt text](Figure_4.png?raw=true "Title")
![Alt text](Figure_5.png?raw=true "Title")

## 📃 License

This project is licensed under the MIT License. See the `LICENSE.txt` file for more information.
