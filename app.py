import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.title("Roller Skating Data Analysis App")

with open("roller_skating_data.xlsx", "rb") as file:
    st.download_button(
        label="Download File",
        data=file,
        file_name="roller_skating_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

uploaded_file = st.file_uploader("Upload your roller_skating_data.xlsx file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.subheader("Raw Data")
    st.write(df.head())

   
    df = df.dropna(axis=1, how='all')
    st.write("Columns:", df.columns)

    
    st.subheader("Clustering Analysis")
    X_cluster = df[['Distance_km','Duration_min','Avg_Speed_kmh','Calories_Burned']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    st.write(df.head())

    # Regression Model
    st.subheader("Calories Prediction Model")
    X = df[['Distance_km','Duration_min','Avg_Speed_kmh','Age','Cluster']]
    y = df['Calories_Burned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = r2_score(y_test, pred)

    st.write(f"Model Accuracy (R²): {accuracy:.2f}")

    # Elbow Method
    st.subheader("Elbow Method")
    inertia = []
    k_range = range(1,10)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(k_range, inertia)
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method For Optimal Clusters")
    st.pyplot(fig1)

    # Scatter Plot
    st.subheader("Cluster Visualization")
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(df['Distance_km'], df['Calories_Burned'], c=df['Cluster'])
    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Calories Burned")
    ax2.set_title("Skater Performance Clusters")
    st.pyplot(fig2)

    # Cluster Count
    st.subheader("Cluster Distribution")
    fig3, ax3 = plt.subplots()
    df['Cluster'].value_counts().plot(kind='bar', ax=ax3)
    ax3.set_title("Number of Skaters in Each Cluster")
    ax3.set_xlabel("Cluster")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)

    # Actual vs Predicted
    st.subheader("Actual vs Predicted")
    fig4, ax4 = plt.subplots()
    ax4.scatter(y_test, pred)
    ax4.set_xlabel("Actual Calories")
    ax4.set_ylabel("Predicted Calories")
    ax4.set_title("Actual vs Predicted Calories Burned")
    st.pyplot(fig4)

    # Feature Importance
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    features = X.columns

    fig5, ax5 = plt.subplots()
    ax5.barh(features, importance)
    ax5.set_title("Feature Importance for Predicting Calories")
    ax5.set_xlabel("Importance")
    ax5.set_ylabel("Features")
    st.pyplot(fig5)

else:
    st.info("Please upload your Excel file to continue.")
