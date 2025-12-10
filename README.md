# ncd_bhutan_project
# Non-Communicable Diseases (NCDs) in Bhutan

## Project Overview
This project analyzes Non-Communicable Diseases (NCDs) in Bhutan using WHO data.
It focuses on understanding trends, performing exploratory data analysis (EDA), building simple prediction models, and creating an interactive Streamlit dashboard.

## Project Workflow and Team Contributions

1. **Data Collection and Project Initialization**  
   - **Done by:** Namgay Wangmo  
   - **Tasks:**  
     - Set up project folder structure (`data/`, `notebooks/`, `src/`, `reports/`, `visuals/`)  
     - Created Python environment and installed dependencies (`requirements.txt`)  
     - Downloaded WHO NCD dataset for Bhutan  

2. **Data Understanding and Initial Cleaning**  
   - **Done by:** Dorji Yangzom  
   - **Tasks:**  
     - Loaded raw CSV (`data/raw/`)  
     - Handled missing values and removed metadata rows  
     - Checked data types and saved cleaned dataset (`data/processed/ncd_clean.csv`)  
     - Documented cleaning steps in `01_data_cleaning.ipynb`  

3. **Advanced Data Cleaning and Feature Preparation**  
   - **Done by:** Chimi Dawa Selden  
   - **Tasks:**  
     - Converted text to numeric columns (YEAR, VALUE)  
     - Dropped rows missing year or numeric value  
     - Prepared dataset for modeling (feature selection and formatting)  

4. **Exploratory Data Analysis (EDA)**  
   - **Done by:** Kinley Tshering  
   - **Tasks:**  
     - Loaded cleaned dataset  
     - Created visualizations (line plots, scatter plots, average trends)  
     - Analyzed sparsity, outliers, and trends  
     - Added explanations in notebook (`02_exploratory_analysis.ipynb`)  

5. **Prediction Model + Streamlit Dashboard (Final Output)**  
   - **Done by:** Bidhan Chhetri  
   - **Tasks:**  
     - Built a simple regression model for prediction  
     - Created interactive Streamlit dashboard (`src/app.py`)  
     - Displayed trends, predicted values, and visualizations  

---

## Folder Structure

