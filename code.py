# import kaggle
# import zipfile

# kaggle datasets download -d uciml/iris
# with zipfile.ZipFile("iris.zip", "r") as zip_ref:
#      zip_ref.extractall("iris")

import sklearn                      # Machine Learning
import pandas as pd                 # Data manipulation
import numpy as np                  # Number operations
import matplotlib.pyplot as plt     # Plotting
import seaborn as sns               # Data visualization

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import metrics

df = pd.read_csv("iris/Iris.csv")

plt.style.use("ggplot")
backgroundColor = "#F2E9E4"

ax = df[df.Species == "Iris-setosa"].plot(kind = "scatter", x = "SepalLengthCm", y = "SepalWidthCm", color = "r", label = "Setosa")
df[df.Species == "Iris-versicolor"].plot(kind = "scatter", x = "SepalLengthCm", y = "SepalWidthCm", color = "g", ax = ax, label = "Versicolor")
df[df.Species == "Iris-virginica"].plot(kind = "scatter", x = "SepalLengthCm", y = "SepalWidthCm", color = "b", ax = ax, label = "Virginica")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Sepal Width", weight = "bold")
fig = ax.get_figure()
fig.patch.set_facecolor(backgroundColor)
plt.show()

ax = df[df.Species == "Iris-setosa"].plot(kind = "scatter", x = "PetalLengthCm", y = "PetalWidthCm", color = "r", label = "Setosa")
df[df.Species == "Iris-versicolor"].plot(kind = "scatter", x = "PetalLengthCm", y = "PetalWidthCm", color = "g", ax = ax, label = "Versicolor")
df[df.Species == "Iris-virginica"].plot(kind = "scatter", x = "PetalLengthCm", y = "PetalWidthCm", color = "b", ax = ax, label = "Virginica")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Petal Length vs Petal Width", weight = "bold")
fig = ax.get_figure()
fig.patch.set_facecolor(backgroundColor)
plt.show()

sns.set_style("whitegrid")
violinPalette = {"Iris-setosa": "#4c72b0", "Iris-versicolor" : "#55a868", "Iris-virginica" : "#c44e52"}
plt.figure(figsize = (15, 10), facecolor = backgroundColor)
plt.subplot(2, 2, 1)
sns.violinplot(x = "Species", hue = "Species", y = "SepalLengthCm", data = df, palette = violinPalette, legend = False)
plt.subplot(2, 2, 2)
sns.violinplot(x = "Species", hue = "Species", y = "SepalWidthCm", data = df, palette = violinPalette, legend = False)
plt.subplot(2, 2, 3)
sns.violinplot(x = "Species", hue = "Species", y = "PetalLengthCm", data = df, palette = violinPalette, legend = False)
plt.subplot(2, 2, 4)
sns.violinplot(x = "Species", hue = "Species", y = "PetalWidthCm", data = df, palette = violinPalette, legend = False)
plt.gcf().suptitle("Iris Features Distribution", weight = "bold", fontsize = 28)
plt.tight_layout()
plt.show()

plt.figure(facecolor=backgroundColor, figsize=(12, 9))
corr_components = df[[
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"]].corr()
sns.heatmap(corr_components, annot=True, fmt=".2f", cmap="vlag", annot_kws={"size": 20})
plt.title("Correlation Between Features", weight = "bold", fontsize = 20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()

# Data Preparation
df.drop('Id',axis=1,inplace=True)
x, y = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]], df.Species
sepal_x, petal_x = df[["SepalLengthCm", "SepalWidthCm"]], df[["PetalLengthCm", "PetalWidthCm"]]

data = [
    ["All Features", x],
    ["Petal", petal_x],
    ["Sepal", sepal_x],
]

from sklearn import svm # Linear SVC, SVC
from sklearn.neighbors import KNeighborsClassifier # KNeighbors Classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier # Random Forest, Gradient Boosting, Voting
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.tree import DecisionTreeClassifier # Decision Tree

import warnings
warnings.filterwarnings('ignore')

models = [
    ["Linear SVC", svm.SVC(kernel = "linear")],
    ["KNeighbors Classifier", KNeighborsClassifier(n_neighbors=3)],
    ["Ensemble Classifier - Random Forest", RandomForestClassifier()],
    ["Ensemble Classifier - Gradient Boosting", GradientBoostingClassifier()],
    ["Ensemble Classifier - Voting", VotingClassifier(estimators=[
        ("Logistic Regression", LogisticRegression()),
        ("svc", svm.SVC(probability=True)),
        ("rf", RandomForestClassifier())
    ], voting = "soft")],
    ["Logistic Regression", LogisticRegression()],
    ["Decision Tree", DecisionTreeClassifier()],
]

result = []

for info in models:
    # print(f"{info[0]} Accuracy:")
    for datainfo in data:
        scorelist = cross_val_score(info[1], datainfo[1], y, cv=5)
        avgscore = scorelist.mean()
        standarddeviation = np.std(scorelist)
        # print(f"{avgscore:.2f} (+/- {standarddeviation:.3f}) [{datainfo[0]}]")
        result.append({
            "model": info[0],
            "feature": datainfo[0],
            "accuracy": avgscore,
            "std": standarddeviation,
        })
    print("")

resultdataframe = pd.DataFrame(result)

plt.figure(figsize=(12, 9), facecolor=backgroundColor)
for feature in resultdataframe["feature"].unique():
    df_feature = resultdataframe[resultdataframe["feature"] == feature]
    x_vals = df_feature["model"]
    y_vals = df_feature["accuracy"]
    y_err = df_feature["std"]

    plt.plot(x_vals, y_vals, marker="o", label=feature)
    plt.fill_between(x_vals, y_vals - y_err, y_vals + y_err, alpha=0.2)
plt.xlabel("ML Model")
plt.xticks(rotation=30)
plt.ylabel("Accuracy + Standard Deviation")
plt.title("Model Accuracy by Feature Set", weight="bold")
plt.legend(title = "Feature")
plt.tight_layout()
plt.show()
print(resultdataframe)
