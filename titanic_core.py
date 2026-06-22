import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load Titanic dataset + clean it
def get_cleaned_df():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv"
    df = pd.read_csv(url)

    df = df.dropna()
    return df

# Plot passenger ages distribution
def get_age_distribution(df: pd.DataFrame):
    age_distribution = plt.figure(figsize=(6, 4))
    sns.histplot(df['Age'], kde=True)
    plt.title('Passenger Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Passenger Count')
    return age_distribution

# Create a scatter plot to show relationship between age and fare
def get_age_fare_scatter(df: pd.DataFrame):
    age_fare_scatter = plt.figure(figsize=(6, 4))
    plt.scatter(df['Age'], df['Fare'], alpha=0.5, color='purple')
    plt.title('Age vs. Fare')
    plt.xlabel('Age')
    plt.ylabel('Fare')
    return age_fare_scatter

# Create a scatter plot to show relationship between class and fare
def get_class_fare_scatter(df: pd.DataFrame):
    class_fare_scatter = plt.figure(figsize=(6, 4))
    plt.title('Class vs. Fare')
    sns.boxplot(x="Pclass", y="Fare", data=df)
    plt.xlabel('Class')
    return class_fare_scatter

# Survivor Analysis Decision Tree
RANDOM_STATE = 42
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
def get_clf(df: pd.DataFrame):
    X = df[features].copy()
    y = df['Survived']

    # Reformat text-based column to numbers for classifier.
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

    # Split data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Initialize and train the Decision Tree
    clf = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # Make predictions and evaluate performance
    y_pred = clf.predict(X_test)

    # Show Accuracy, Report, and Tree Graph
    clf_accuracy = accuracy_score(y_test, y_pred)

    clf_report = classification_report(y_test, y_pred)

    decision_tree_graph = plt.figure(figsize=(6, 4))
    plot_tree(clf)

    return clf, clf_accuracy, clf_report, decision_tree_graph

if __name__ == '__main__':
    df = get_cleaned_df()
    get_age_distribution(df)
    get_age_fare_scatter(df)
    get_class_fare_scatter(df)
    clf, clf_accuracy, clf_report, _ = get_clf(df)
    print("Decision Tree Classifier")
    print(f"Test Accuracy: {clf_accuracy:.2%}\n")
    print("Classification Report:")
    print(clf_report)
    plt.show()