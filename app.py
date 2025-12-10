
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Global Health Statistics Predictor (Manual Implementation)")

@st.cache_resource
def load_and_prep_data():
    
    df = pd.read_csv("cleaned_data.csv")
    
    
    df['DIMENSION_NAME'] = df['DIMENSION_NAME'].fillna('Total')

    
    country_map = {val: i for i, val in enumerate(df['COUNTRY_DISPLAY'].unique())}
    gho_map = {val: i for i, val in enumerate(df['GHO_DISPLAY'].unique())}
    dim_map = {val: i for i, val in enumerate(df['DIMENSION_NAME'].unique())}

    
    df['country_encoded'] = df['COUNTRY_DISPLAY'].map(country_map)
    df['gho_encoded'] = df['GHO_DISPLAY'].map(gho_map)
    df['dim_encoded'] = df['DIMENSION_NAME'].map(dim_map)

    
    return df, country_map, gho_map, dim_map

@st.cache_resource
def train_model(df):
    
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    split_index = int(len(df_shuffled) * 0.8)
    
    
    train_data = df_shuffled.iloc[:split_index]
    test_data = df_shuffled.iloc[split_index:]
    
    
    feature_cols = ['country_encoded', 'gho_encoded', 'dim_encoded', 'YEAR_DISPLAY']
    X_train = train_data[feature_cols]
    y_train = train_data['Value_num']
    X_test = test_data[feature_cols]
    y_test = test_data['Value_num']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    
    predictions = model.predict(X_test)
    
    ss_res = np.sum((y_test - predictions) ** 2)      
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)    
    r2_score = 1 - (ss_res / ss_tot)
    
    return model, r2_score

df, country_map, gho_map, dim_map = load_and_prep_data()
model, accuracy = train_model(df)



menu = st.sidebar.selectbox(
    "Navigate",
    ["Dataset Overview", "Visualizations", "Train Model Summary", "Predict Health Value"]
)

if menu == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("First 5 rows of the cleaned data:")
    st.dataframe(df[['COUNTRY_DISPLAY', 'YEAR_DISPLAY', 'GHO_DISPLAY', 'DIMENSION_NAME', 'Value_num']].head())
    
    st.header("Summary of Data")
    st.dataframe(df.describe())
    
    st.header("Distribution of Target Value")
    st.bar_chart(df['Value_num'].value_counts().head(20))

elif menu == "Visualizations":
    st.header("Visualizations")
    viz_type = st.selectbox("Choose chart type:", ["Correlation Heatmap", "Line Chart", "Bar Chart", "Scatter Plot"])
    
    if viz_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        corr_cols = ['country_encoded', 'gho_encoded', 'dim_encoded', 'YEAR_DISPLAY', 'Value_num']
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)
        
    elif viz_type == "Line Chart":
        st.subheader("Average Value Trend over Years")
        st.line_chart(df.groupby('YEAR_DISPLAY')['Value_num'].mean())
        
    elif viz_type == "Bar Chart":
        st.subheader("Top 10 Countries by Average Value")
        st.bar_chart(df.groupby('COUNTRY_DISPLAY')['Value_num'].mean().sort_values(ascending=False).head(10))
        
    elif viz_type == "Scatter Plot":
        st.subheader("Year vs Value")
        plt.figure(figsize=(10, 5))
        plt.scatter(df['YEAR_DISPLAY'], df['Value_num'], alpha=0.5)
        st.pyplot(plt)

elif menu == "Train Model Summary":
    st.header("Model Training Summary")
    st.metric(label="Model Accuracy (R2 Score)", value=f"{accuracy:.4f}")
    
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": ['Country', 'Indicator (GHO)', 'Dimension', 'Year'],
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    st.bar_chart(importance_df.set_index("Feature"))

elif menu == "Predict Health Value":
    st.header("Predict Health Indicator Value")
    
    # User inputs
    selected_country = st.selectbox("Select Country", list(country_map.keys()))
    selected_indicator = st.selectbox("Select Health Indicator", list(gho_map.keys()))
    selected_dimension = st.selectbox("Select Dimension", list(dim_map.keys()))
    selected_year = st.slider("Select Year", int(df['YEAR_DISPLAY'].min()), int(df['YEAR_DISPLAY'].max()), 2022)
    
    if st.button("Predict"):
        try:
            
            c_val = country_map[selected_country]
            g_val = gho_map[selected_indicator]
            d_val = dim_map[selected_dimension]
            
            
            features = np.array([[c_val, g_val, d_val, selected_year]])
            prediction = model.predict(features)[0]
            
            # --- START: Added Descriptive Text ---
            st.success(f"Predicted Value: **{prediction:.2f}**")
            
            st.info(
                f"This predicted value represents the expected measurement for the health indicator: "
                f"**'{selected_indicator}'** "
                f"for the **{selected_dimension}** population in **{selected_country}** in the year **{selected_year}**."
            )
            # --- END: Added Descriptive Text ---

        except Exception as e:
            st.error(f"Prediction Error: {e}")
