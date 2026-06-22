import streamlit as st
from titanic_core import (
    get_cleaned_df,
    get_age_distribution,
    get_age_fare_scatter,
    get_class_fare_scatter,
    get_clf
)

st.set_page_config(
    page_title="Titanic Survivor Data Machine Learning",
    layout="wide"
)

# Title
"# Titanic Survivor Data Machine Learning"

# Cleaned Data
"## Cleaned Data"
df = st.cache_data(get_cleaned_df)()
df

# Graphs
"## Graphs"
st.write(st.cache_data(get_age_distribution)(df))
st.write(st.cache_data(get_age_fare_scatter)(df))
st.write(st.cache_resource(get_class_fare_scatter)(df))

# Decision Tree
get_clf_cached = st.cache_data(get_clf)
clf, clf_accuracy, clf_report, decision_tree_graph = get_clf_cached(df)
"## Decision Tree"
f"**Accuracy Rate:** {clf_accuracy:.2%}"
f"""**Classification Report:**
```
{clf_report}
```"""
decision_tree_graph

with st.form("clf_form"):
    "### Run Decision Tree Classifier"
    p_class = st.number_input("Passenger Class (1, 2, or 3)", min_value=1, max_value=3)
    p_sex = st.number_input("Sex (1 for Female, 0 for Male)", min_value=0, max_value=1)
    p_age = st.number_input("Age", min_value=0)
    p_sibsp = st.number_input("Siblings + Spouses On Board", min_value=0)
    p_parch = st.number_input("Parents + Children On Board", min_value=0)
    p_fare = st.number_input("Ticket Fare", min_value=0)
    clf_form_submit_btn = st.form_submit_button(label="Run Classifier")

if clf_form_submit_btn:
    data = [[p_class, p_sex, p_age, p_sibsp, p_parch, p_fare]]
    result = clf.predict(data)[0]
    prob = clf.predict_proba(data)[:, result][0] * 100
    if result:
        st.success(f"**Prediction:** Passenger Survived (Confidence: {prob:.2f}%)")
    else:
        st.error(f"**Prediction:** Passenger Did Not Survive (Confidence: {prob:.2f}%)")
