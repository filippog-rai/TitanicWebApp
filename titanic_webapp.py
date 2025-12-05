import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import kagglehub
import base64

from itertools import product

# Stampa la directory corrente
st.write(f"Directory Corrente: {os.getcwd()}")

# Stampa i file presenti nella cartella radice (dovresti vedere 'app.py' e 'asset/')
st.write("File nella root:", os.listdir()) 

# Prova a listare i file nella cartella 'asset' (ATTENZIONE alla case-sensitivity)
try:
    st.write(f"File nella cartella asset: {os.listdir('asset')}")
except FileNotFoundError:
    st.write("Errore: la cartella 'asset' non è stata trovata o ha problemi di permessi.")
    
def set_background(image_file):
    """
    Background image.
    """
    try:
        # backround file
        with open(image_file, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()
        
        
        css_string = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        [data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.7); /* Esempio: Sfondo bianco semi-trasparente */
        }}
        </style>
        """
        
        
        st.markdown(css_string, unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.error(f"Error: Image File '{image_file}' not found. Make sure it's in the same dir of the this file.")
    except Exception as e:
        st.error(f"Error during background setting: {e}")

set_background('assets/titanic_background.png')

st.markdown("""
<style>
/* stRadio */
.stRadio > label {
    background-color: rgba(255, 255, 255, 0.1);
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* stSelectbox */
.stSelectbox > label {
    background-color: rgba(255, 255, 255, 0.1);
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* stColumns */
.stColumns > label {
    background-color: rgba(255, 255, 255, 0.1);
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    font-size: 3rem;
}
div[data-testid="stMetricLabel"] {
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.stApp {
    padding-top: 0px !important;
}

[data-testid="stAppViewContainer"] {
    padding-top: 0rem; 
    margin-top: -100px;
}


h1 {
    margin-top: 2rem;
}

body {
    margin: 0 !important;
    padding: 0 !important;
}

</style>
""", unsafe_allow_html=True)


# Import data from kaggle
path = kagglehub.dataset_download("yasserh/titanic-dataset")

df = pd.read_csv(os.path.join(path, "Titanic-Dataset.csv"))

# --- 1. Data Simulation ---

# Drops columns with too many missing values
COLUMN_WITH_TOO_MANY_MISSING_VALUES = ["Cabin"]
df.drop(COLUMN_WITH_TOO_MANY_MISSING_VALUES, axis=1, inplace=True, errors="ignore")
# Fill null values of age column with median
df.fillna({"Age": df["Age"].median()}, inplace=True)
# Fill null values of Embarked with most frequent value
df.fillna({"Embarked": df["Embarked"].mode()[0]}, inplace=True)
# Drop columns that are mostly identifiers
COLUMN_MOSTLY_IDENTIFIERS = ["PassengerId", "Name", "Ticket"]
df.drop(COLUMN_MOSTLY_IDENTIFIERS, inplace=True, axis=1, errors="ignore")
# Create 'FamilySize' and 'IsAlone'
df['FamilySize'] = df['SibSp'] + df['Parch']

# Create age groups
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0, 18, 35, 50, 100],
    labels=['Child','YoungAdult','Adult','Senior']
)

ALL_OPTION = 'All'

def get_filter_options(column):
    return [ALL_OPTION] + sorted(df[column].dropna().unique().tolist())

st.markdown("<p style='font-size: 18px; color: grey;'>RAI-7002 | Assessment 3 | Group 4</p>", unsafe_allow_html=True)

# --- 2. STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Titanic: Descriptive Analysis", page_icon="⚓")
st.title("Survivor Analysis on the Titanic Dataset")

# Definisci i valori iniziali (la prima opzione, che è 'Tutti')
DEFAULT_SEX = ALL_OPTION
DEFAULT_PCLASS = ALL_OPTION
DEFAULT_AGEGROUP = ALL_OPTION

def reset_filters():
    st.session_state.selected_sex = DEFAULT_SEX
    st.session_state.selected_age = DEFAULT_AGEGROUP
    st.session_state.selected_pclass = DEFAULT_PCLASS
    st.session_state.submitted = False

st.button("Reset all filters", on_click=reset_filters)

with st.form("filter_form"):
    st.markdown("Select passenger profile to identify survival probability.")

    if 'selected_sex' not in st.session_state:
        st.session_state.selected_sex = DEFAULT_SEX
    if 'selected_age' not in st.session_state:
        st.session_state.selected_age = DEFAULT_SEX
    if 'selected_pclass' not in st.session_state:
        st.session_state.selected_pclass = DEFAULT_SEX

    cols = st.columns(2)

    with cols[0]:
        st.subheader("Demographic Factors")
        selected_sex = st.selectbox("**Sex**", options=df['Sex'].unique())
        selected_age = st.selectbox("**Age Group**", options=df['AgeGroup'].unique())

    with cols[1]:
        st.subheader("Socio-Economic Selector")
        selected_pclass = st.selectbox("**Class**", options=sorted(df['Pclass'].unique()))


    submitted = st.form_submit_button("Apply Filters")
    st.session_state.submitted = submitted

st.divider()
df_for_3d = df.copy()
# --- 3. NEUTRAL FILTER CREATION ---
A = df['Sex'].unique()
B = df['AgeGroup'].unique()
C = df['Pclass'].unique()
combination = list(product(A, B, C))
df_filter_pivot = pd.DataFrame(combination, columns=['Sex', 'AgeGroup', 'Pclass'])
df_filter_pivot['FilterValue'] = 1

# --- 4. DATAFRAME FILTER ---
if st.session_state.submitted:

    st.success(f"{selected_sex} - {selected_age} - {selected_pclass} selected.")

    # 4b. Final filter
    if selected_sex != ALL_OPTION and selected_pclass != ALL_OPTION and selected_age != ALL_OPTION:
        df_filter_final = (df['Sex'] == selected_sex) & (df['Pclass'] == selected_pclass) & (df['AgeGroup'] == selected_age)
        df_filter_pivot_int = (df_filter_pivot['Sex'] == selected_sex) & (df_filter_pivot['Pclass'] == selected_pclass) & (df_filter_pivot['AgeGroup'] == selected_age)
        filtered_df = df[df_filter_final]

    df_filter_pivot['FilterValue'] = np.where(df_filter_pivot_int, 1, 0)

    # --- 5. Final Evaluation---
    total = len(filtered_df)
    st.header("Survival Probability and Count distribution")

    if total == 0:
        st.warning("**ATTENTION:** No passengers found with these characteristics.")
        
    else:
        survived_count = filtered_df['Survived'].sum()
        not_survived_count = total - survived_count
        survival_prob = (survived_count / total) * 100
        
        col_main_1, col_chart = st.columns([1, 2])
        
        with col_main_1:
            #st.header("Survival Probability")
            metric_container = st.empty()

            final_prob = survival_prob 
            animation_duration = 1.0
            update_interval = 0.02
            
            # 3. Animation
            num_steps = int(animation_duration / update_interval)
            
            for step in range(num_steps + 1):
                progress = step / num_steps
                if progress > 1.0: progress = 1.0
                    
                current_prob = final_prob * progress
                
                metric_container.metric(
                    label="",#"Survival probability",
                    value=f"{current_prob:.1f}%",
                    delta=f"{survived_count} survived out of {total} total"
                )
                
                time.sleep(update_interval)

            prob = survival_prob
            if prob >= 70:
                status_icon = "✅"
                status_title = "High chance to survive"
                st.success(f"{status_icon} **{status_title}**")
            
            elif prob >= 40:
                status_icon = "⚖️"
                status_title = "Uncertain Outcome"
                st.info(f"{status_icon} **{status_title}**")
            
            else:
                status_icon = "❌"
                status_title = "Low chance to survive"
                st.error(f"{status_icon} **{status_title}**")

        with col_chart:
            counts_df = pd.DataFrame({
                'Status': ['Survived', 'Not Survived'],
                'Count': [survived_count, not_survived_count]
            })
            
            import plotly.express as px
            fig = px.bar(
                counts_df, 
                x='Status', 
                y='Count', 
                color='Status',
                color_discrete_map={'Survived': 'green', 'Not Survived': 'red'}#,
                #title="Survivor Count Distribution"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Press 'Apply Filter' to see outcome.")

# --- 6. STORYTELLING  (3D scatterplot) ---
st.markdown("---")
st.header("Focus on top 3 predictors")
st.subheader("Sex / Pclass / Age Group")


df_agg_3d = df_for_3d.groupby(['Pclass', 'Sex', 'AgeGroup'], observed=True).agg(
SurvivalRate_dec=('Survived', 'mean'),
TotalCount=('Fare', 'size')
).reset_index()
df_agg_3d['SurvivalRate_int'] = round(df_agg_3d['SurvivalRate_dec'] * 100,2)
sex_map = {0: 'female', 1:'male'}
df_agg_3d['Sex_map'] = df_agg_3d['Sex'].map(sex_map)
age_map = {'Child': 1, 'YoungAdult': 2, 'Adult': 3, 'Senior': 4}
df_agg_3d['AgeGroup_Num'] = df_agg_3d['AgeGroup'].map(age_map)
df_agg_3d.rename(columns={'TotalCount':'TotalCount_int'}, inplace=True)

## Filtering before 3D plot
df_union = pd.merge(
    df_agg_3d,
    df_filter_pivot[['Sex', 'AgeGroup', 'Pclass', 'FilterValue']],
    on=['Sex', 'AgeGroup', 'Pclass'],
    how='left'
)
df_union['Survival_Rate'] = df_union['SurvivalRate_int'] * df_union['FilterValue']
df_union['Total_Count'] = df_union['TotalCount_int'] * df_union['FilterValue']


fig_3d = px.scatter_3d(
    df_union,
    x='Pclass', 
    y='Sex', 
    z='AgeGroup_Num',
    color='Survival_Rate',
    size='Total_Count',
    size_max=50,
    height=600,
    color_continuous_scale=px.colors.diverging.RdYlGn,
    range_color=[0, 100]
)
# Pclass (X-axis)
fig_3d.update_layout(scene=dict(
    xaxis=dict(
    title='Pclass'
),
# Sex (Y-axis)
yaxis=dict(
    tickvals=list(sex_map.keys()),
    ticktext=list(sex_map.values()),
    title='Sex'
),
# AgeGroup (Z-axis)
zaxis=dict(
    tickvals=list(age_map.values()),
    ticktext=list(age_map.keys()),
    title='Age Group'
)
))

st.plotly_chart(fig_3d, use_container_width=True)

