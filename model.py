import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("GalaxyZoo1_DR_table2.csv")

# Limit to 5000 rows
df = df.head(5000)

# Define target class
def classify(row):
    if row['SPIRAL'] == 1:
        return 'spiral'
    elif row['ELLIPTICAL'] == 1:
        return 'elliptical'
    else:
        return 'uncertain'

df['morphology_class'] = df.apply(classify, axis=1)

# Feature columns
features = ['P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK', 'P_MG', 'P_CS', 'P_EL_DEBIASED', 'P_CS_DEBIASED']
X = df[features]
y = LabelEncoder().fit_transform(df['morphology_class'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

# Confusion matrix plot
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['elliptical', 'spiral', 'uncertain'],
            yticklabels=['elliptical', 'spiral', 'uncertain'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
