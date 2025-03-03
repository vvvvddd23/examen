import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Titlu și descriere
st.title("Clasificarea Cancerului Mamar - Breast Cancer Wisconsin (Original)")
st.write("Această aplicație clasifică cazurile de cancer mamar în funcție de recurență (no-recurrence-events vs. recurrence-events) pe baza caracteristicilor clinice.")

# 1. Încărcarea datelor
@st.cache_data
def load_data():
    columns = ["Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"]
    df = pd.read_csv("breast-cancer.data", names=columns, na_values="?")
    return df

df = load_data()
st.subheader("Datele Încărcate")
st.write(df.head())
st.write(f"Dimensiune set de date: {df.shape}")

# 2. Preprocesarea datelor
def preprocess_data(df):
    df = df.fillna(df.mode().iloc[0])  # Imputare cu valoarea modală
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == "object":
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    X = df.drop("Class", axis=1)
    y = df["Class"]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, label_encoders, df

X, y, label_encoders, df = preprocess_data(df)

# 3. Împărțirea datelor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Definirea și antrenarea modelelor
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Linear Regression": LinearRegression()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    if name == "Linear Regression":
        y_pred = np.where(model.predict(X_test) > 0.5, 1, 0)
    else:
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=["No Recurrence", "Recurrence"])
    
    results[name] = {
        "accuracy": accuracy,
        "conf_matrix": conf_matrix,
        "class_report": class_report,
        "y_pred": y_pred
    }

# 5. Afișarea rezultatelor modelelor
st.subheader("Rezultatele Modelelor")
for name, result in results.items():
    st.write(f"### {name}")
    st.write(f"Acuratețe: {result['accuracy']:.2f}")
    st.text("Matrice de Confuzie:")
    st.write(result['conf_matrix'])
    st.text("Raport de Clasificare:")
    st.write(result['class_report'])

# 6. Grafice
st.subheader("Vizualizări")

# Grafic 1: Matrice de confuzie pentru fiecare model
for name, result in results.items():
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(result['conf_matrix'], annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No Recurrence", "Recurrence"], yticklabels=["No Recurrence", "Recurrence"], ax=ax)
    ax.set_title(f"Matrice de Confuzie - {name}")
    ax.set_xlabel("Predicție")
    ax.set_ylabel("Adevărat")
    st.pyplot(fig)

# Grafic 2: Importanța caracteristicilor pentru Random Forest
importances_rf = models["Random Forest"].feature_importances_
coef_df_rf = pd.DataFrame({"Feature": df.drop("Class", axis=1).columns, "Importance": importances_rf})
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=coef_df_rf.sort_values(by="Importance", ascending=False), ax=ax3)
ax3.set_title("Importanța Caracteristicilor - Random Forest")
ax3.axvline(x=0, color='black', linestyle='--')
st.pyplot(fig3)
# Grafic 3: Precizie în funcție de vârstă
st.subheader("Precizie în funcție de vârstă")
if "age" in df.columns:
    df["age"] = label_encoders["age"].inverse_transform(df["age"])
    age_groups = df.groupby("age")["Class"].mean()
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=age_groups.index, y=age_groups.values, marker='o', ax=ax4)
    ax4.set_title("Precizie în funcție de vârstă")
    ax4.set_xlabel("Vârstă")
    ax4.set_ylabel("Proporție Recurrence")
    st.pyplot(fig4)

# Grafic 4: Curbe Precision-Recall
st.subheader("Curbe Precision-Recall")
plt.figure(figsize=(10, 6))
for name, model in models.items():
    if name == "Linear Regression":
        y_prob = np.clip(model.predict(X_test), 0, 1)
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.plot(recall, precision, label=f'{name} (AP = {ap:.2f})')
plt.title("Curbe Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
st.pyplot(plt)
# 7. Comparație între metode
st.subheader("Comparație între Metode")
accuracies = [results[name]["accuracy"] for name in results]
fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=accuracies, ax=ax_comp)
ax_comp.set_title("Comparație Acuratețe Modele")
ax_comp.set_ylabel("Acuratețe")
ax_comp.set_xlabel("Model")
for i, v in enumerate(accuracies):
    ax_comp.text(i, v + 0.01, f"{v:.2f}", ha="center")
st.pyplot(fig_comp)