import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("Financial_inclusion_dataset.csv")

# Nettoyage
df.drop_duplicates(inplace=True)
df.fillna(method="ffill", inplace=True)

# Encodage
encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("bank_account", axis=1)
y = df["bank_account"]

# Sauvegarde des colonnes
joblib.dump(X.columns.tolist(), "features.pkl")

# Entraînement
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Accuracy :", accuracy_score(y_test, model.predict(X_test)))

# Sauvegardes
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")
