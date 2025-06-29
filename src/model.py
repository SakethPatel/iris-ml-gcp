from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
#commented
def train_model(df):
    # Separate features and target
    X = df.iloc[:, :-1]  # Features (all columns except species)
    y = LabelEncoder().fit_transform(df["species"])  # Encode target labels

    # Train model
    model = DecisionTreeClassifier()
    model.fit(X, y)

    # Predict on training data
    predictions = model.predict(X)

    # Calculate accuracy
    acc = accuracy_score(y, predictions)

    return model, acc
