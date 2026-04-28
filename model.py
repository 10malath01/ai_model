import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = "model.pkl"

def train_model(df):
    X = df["text"]
    y = df["compliance_score"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("rf", RandomForestRegressor(
            n_estimators=150,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    return joblib.load(MODEL_PATH)
