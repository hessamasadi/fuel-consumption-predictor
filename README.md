# Vehicle Fuel Consumption Predictor
An end-to-end Machine Learning web application for predicting vehicle fuel consumption (L/100 km) and estimating real-world trip costs. Built with Streamlit and powered by Random Forest Regressors, the app allows users to either input global vehicle specifications manually or select from a pre-loaded database of popular Iranian cars.

# Key Features
Three Specialized ML Models
Instead of a single model, this app uses three separate Random Forest models trained specifically for City, Highway, and Combined driving conditions, achieving an outstanding R² score of > 0.90 across all three.

# Localized Iranian Car Database
Includes a dedicated, fully translated (Persian) section for popular domestic vehicles (e.g., Peugeot Pars, Samand, Dena, Tara, Shahin), mapping their mechanical specs to global equivalents for accurate ML inference.

# Smart Trip Cost Calculator
Features a unique cost-estimation engine tailored to Iran's real-world 3-tier smart fuel card quota system (calculating costs dynamically across the 1,500T, 3,000T, and 5,000T Toman pricing tiers based on the user's monthly quota usage).

# Robust Data Cleaning Pipeline
Implements a rigorous preprocessing pipeline using Regex to standardize messy categorical data (e.g., unifying inconsistent vehicle class punctuations and casings) before One-Hot Encoding.

# Bilingual UI
Seamlessly switches between a Persian interface (for local cars and cost calculations) and an English interface (for global manual selections).
