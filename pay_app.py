import streamlit as st
import sklearn
import toml

# Load custom theme configuration
with open("custom_theme.toml", "r") as theme_file:
    custom_theme = toml.load(theme_file)

# Extract the 'theme' dictionary from the loaded configuration
theme_settings = custom_theme.get("theme", {})

# Apply the custom theme using **kwargs
st.set_page_config(
    page_title="techie-pay",
    page_icon="Business-Technology-Digital-High-Tech-World-Background.png",
    layout="centered",
    initial_sidebar_state="collapsed",
    **theme_settings
)
from predict_pg import show_predict_pg


show_predict_pg()
