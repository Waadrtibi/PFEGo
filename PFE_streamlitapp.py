import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.title("🚗 Déploiement et Analyse du Modèle ML - Stations Recharge Véhicules Électriques")

# --- Chargement des données ---
DATA_PATH = r"C:\Users\Waad RTIBI\pfeGo\detailed_ev_charging_stations.csv"
st.markdown("### 1. Chargement et Aperçu des données")
df = pd.read_csv(DATA_PATH)
st.dataframe(df.head())

# Nettoyage
df_clean = df.copy()
df_clean = df_clean.drop(columns=["Station ID", "Address"], errors='ignore')

# Conversion colonnes numériques
num_cols = ["Usage Stats (avg users/day)", "Latitude", "Longitude",
            "Distance to City (km)", "Charging Capacity (kW)",
            "Installation Year", "Reviews (Rating)", "Parking Spots", "Cost (USD/kWh)"]
for col in num_cols:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Traitement colonnes catégoriques et booléens
if "Availability" in df_clean.columns:
    df_clean["Availability_24h"] = df_clean["Availability"].apply(lambda x: 1 if isinstance(x, str) and "24/7" in x else 0)
    df_clean = df_clean.drop(columns=["Availability"])
if "Renewable Energy Source" in df_clean.columns:
    df_clean["Renewable Energy Source"] = df_clean["Renewable Energy Source"].map({"Yes": 1, "No": 0})
freq_map = {"Monthly": 2, "Annually": 1}
if "Maintenance Frequency" in df_clean.columns:
    df_clean["Maintenance Frequency"] = df_clean["Maintenance Frequency"].map(freq_map).fillna(0)

df_clean = df_clean.dropna()

# --- Préparation données pour ML ---
st.markdown("### 2. Préparation des données pour le modèle")
target = "Usage Stats (avg users/day)"
X = df_clean.drop(columns=[target])
y = df_clean[target]

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

st.write(f"Variables numériques : {num_features}")
st.write(f"Variables catégoriques : {cat_features}")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Entraînement du modèle ---
st.markdown("### 3. Entraînement du modèle ElasticNetCV")

elastic_cv = ElasticNetCV(
    l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
    alphas=np.logspace(-4, 0, 50),
    cv=5,
    random_state=42,
    max_iter=10000
)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", elastic_cv)
])

if st.button("▶️ Entraîner le modèle"):
    with st.spinner("Entraînement en cours..."):
        pipe.fit(X_train, y_train)
    st.success("Modèle entraîné avec succès !")

    # Prédictions et métriques
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    def print_metrics(y_true, y_pred, dataset="Test"):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        st.write(f"**{dataset} set metrics :**")
        st.write(f"- R² = {r2:.4f}")
        st.write(f"- MAE = {mae:.4f}")
        st.write(f"- RMSE = {rmse:.4f}")

    print_metrics(y_train, y_train_pred, "Train")
    print_metrics(y_test, y_test_pred, "Test")

    st.write(f"Meilleur alpha (régularisation) : {elastic_cv.alpha_:.6f}")
    st.write(f"Meilleur l1_ratio (mix L1/L2) : {elastic_cv.l1_ratio_}")

    # Sauvegarde du modèle
    with open("model_ev_elasticnet.pkl", "wb") as f:
        pickle.dump(pipe, f)
    st.info("Modèle sauvegardé localement sous 'model_ev_elasticnet.pkl'.")

# --- Prédiction utilisateur ---
st.markdown("### 4. Prédiction interactive")

input_data = {}
for col in num_features:
    val = st.number_input(f"{col}", value=float(df_clean[col].median()), format="%.2f")
    input_data[col] = val

for col in cat_features:
    options = list(df_clean[col].unique())
    val = st.selectbox(f"{col}", options)
    input_data[col] = val

if st.button("🔮 Prédire l'usage moyen"):
    try:
        with open("model_ev_elasticnet.pkl", "rb") as f:
            model_loaded = pickle.load(f)
        df_input = pd.DataFrame([input_data])
        pred = model_loaded.predict(df_input)[0]
        st.success(f"👉 Usage moyen prédit : {pred:.2f} utilisateurs/jour")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        st.info("Veuillez entraîner d'abord le modèle en cliquant sur 'Entraîner le modèle'.")

# --- Visualisations ---
st.header("📊 Visualisations des données")

st.subheader("Coût selon la source d’énergie renouvelable")
fig1, ax1 = plt.subplots()
sns.boxplot(data=df_clean, x='Renewable Energy Source', y='Cost (USD/kWh)', ax=ax1)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Non renouvelable", "Renouvelable"])
st.pyplot(fig1)

st.subheader("Carte des stations EV par type de chargeur")
fig2 = px.scatter_geo(df_clean,
                     lat='Latitude',
                     lon='Longitude',
                     color='Charger Type',
                     title='Localisation mondiale des stations EV',
                     hover_name='Station Operator')
st.plotly_chart(fig2)

st.subheader("Heatmap corrélation variables numériques")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(df_clean.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

st.subheader("Répartition des types de chargeurs")
fig4, ax4 = plt.subplots()
sns.countplot(data=df_clean, x='Charger Type', ax=ax4)
plt.xticks(rotation=45)
st.pyplot(fig4)
