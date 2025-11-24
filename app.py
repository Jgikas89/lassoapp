import streamlit as st
import mglearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Set page config
st.set_page_config(page_title="Lasso Regression Analysis", layout="wide")

# Title and Introduction
st.title("Lasso Regression on Boston Housing Data")
st.markdown("""
This application demonstrates **Lasso Regression** on the extended Boston Housing dataset. 
Lasso (Least Absolute Shrinkage and Selection Operator) is a linear model that estimates sparse coefficients. 
It is useful for high-dimensional data as it can force some coefficients to be exactly zero, effectively performing feature selection.
""")

# Load Data
@st.cache_data
def load_data():
    X, y = mglearn.datasets.load_extended_boston()
    return X, y

X, y = load_data()

st.sidebar.header("Model Parameters")
st.sidebar.markdown("Adjust the regularization strength (Alpha).")

# Alpha Slider (linear scale 0 to 2)
alpha = st.sidebar.slider("Alpha (Regularization Strength)", 0.00001, 2.0, 1.0, 0.01)
st.sidebar.write(f"Current Alpha: **{alpha:.5f}**")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)

# Scale Data (Crucial for Lasso)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
lasso = Lasso(alpha=alpha, max_iter=100000).fit(X_train, y_train)

# Metrics
train_score = lasso.score(X_train, y_train)
test_score = lasso.score(X_test, y_test)
num_features = np.sum(lasso.coef_ != 0)

# Display Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Training R²", f"{train_score:.4f}")
col2.metric("Test R²", f"{test_score:.4f}")
col3.metric("Features Used", f"{num_features} / {X.shape[1]}")

st.divider()

# Explanation
st.info("""
**What is happening?**
*   **Alpha**: Controls the strength of regularization. 
    *   **High Alpha**: More regularization. More coefficients become zero. Model becomes simpler (potentially underfitting).
    *   **Low Alpha**: Less regularization. Model becomes more complex (resembles standard Linear Regression, potentially overfitting).
*   **Lasso**: Tries to minimize the error + alpha * sum of absolute values of coefficients.
""")

# Visualization 1: Coefficients
st.subheader("Lasso Coefficients")
st.markdown("Magnitude of coefficients for the current model. Zero coefficients are features removed by Lasso.")

fig_coef, ax_coef = plt.subplots(figsize=(10, 4))
ax_coef.plot(lasso.coef_, 'o', label="Lasso coefficients")
ax_coef.set_xlabel("Feature Index")
ax_coef.set_ylabel("Coefficient Magnitude")
ax_coef.set_title(f"Coefficients for alpha={alpha:.5f}")
ax_coef.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax_coef.grid(True, alpha=0.3)
st.pyplot(fig_coef)


# Non-zero Coefficients List
st.subheader("Selected Predictors (Non-Zero Coefficients)")
st.markdown("Features with non-zero coefficients are the ones selected by the Lasso model.")

# Generate feature names
original_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
poly = PolynomialFeatures(degree=2, include_bias=False)
# Fit on dummy data to get feature names
poly.fit(np.zeros((1, 13)))
feature_names = poly.get_feature_names_out(original_features)

# Create DataFrame
coef_df = pd.DataFrame({
    'Feature Index': feature_names,
    'Coefficient Value': lasso.coef_,
    'Absolute Coefficient Value': np.abs(lasso.coef_)
})

# Filter non-zero
non_zero_df = coef_df[coef_df['Coefficient Value'] != 0].sort_values(by='Absolute Coefficient Value', ascending=False)

st.write(f"**{len(non_zero_df)} predictors selected** out of 104 features.")
st.dataframe(non_zero_df, use_container_width=True)

st.subheader("Regularization Path")
st.markdown("How coefficients change as Alpha changes. The vertical line represents the currently selected Alpha.")

@st.cache_data
def compute_regularization_path(X, y):
    # Linear space range for alpha to match slider better, plus logspace for small values
    # Combining ranges to cover small alphas (where interesting things happen) up to 2
    alphas = np.concatenate([
        np.logspace(-5, -1, 20),
        np.linspace(0.1, 2, 50)
    ])
    alphas = np.sort(np.unique(alphas))
    
    coefs = []
    train_scores = []
    test_scores = []
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=100)
    
    # Scale within the path function as well
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    for a in alphas:
        # Increase max_iter for convergence with small alphas
        lasso_iter = Lasso(alpha=a, max_iter=100000)
        lasso_iter.fit(X_tr, y_tr)
        coefs.append(lasso_iter.coef_)
        train_scores.append(lasso_iter.score(X_tr, y_tr))
        test_scores.append(lasso_iter.score(X_te, y_te))
        
    return alphas, np.array(coefs), train_scores, test_scores

path_alphas, path_coefs, path_train_scores, path_test_scores = compute_regularization_path(X, y)

fig_path, ax_path = plt.subplots(figsize=(10, 6))
# Plot all coefficients
for i in range(path_coefs.shape[1]):
    ax_path.plot(path_alphas, path_coefs[:, i], alpha=0.5, linewidth=0.8)

ax_path.set_xscale('linear')
ax_path.set_xlim(0, 2)
ax_path.set_xlabel('Alpha')
ax_path.set_ylabel('Coefficients')
ax_path.set_title('Lasso Coefficients Path')
ax_path.axvline(alpha, color='red', linestyle='--', label=f'Current Alpha: {alpha:.5f}')
ax_path.legend(loc='upper right')
ax_path.grid(True, alpha=0.3)

st.pyplot(fig_path)

# Performance vs Alpha
st.subheader("Model Performance vs Alpha")
fig_perf, ax_perf = plt.subplots(figsize=(10, 4))
ax_perf.plot(path_alphas, path_train_scores, label='Training R²')
ax_perf.plot(path_alphas, path_test_scores, label='Test R²')
ax_perf.set_xscale('linear')
ax_perf.set_xlim(0, 2)
ax_perf.set_xlabel('Alpha')
ax_perf.set_ylabel('R² Score')
ax_perf.axvline(alpha, color='red', linestyle='--', label=f'Current Alpha')
ax_perf.legend()
ax_perf.grid(True, alpha=0.3)
st.pyplot(fig_perf)


