import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("data/covid_toy.csv")

df.head()

df.isnull().sum()

lb=LabelEncoder()
df['gender']=lb.fit_transform(df['gender'])
df['cough']=lb.fit_transform(df['cough'])
df['city']=lb.fit_transform(df['city'])
df['has_covid']=lb.fit_transform(df['has_covid'])
sc=SimpleImputer(strategy='mean')
df=pd.DataFrame(sc.fit_transform(df),columns=df.columns)
X = df.drop("has_covid",axis=1)
y = df["has_covid"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model in the correct directory
joblib.dump(model, "api/model.pkl")
print("Model saved successfully!")


