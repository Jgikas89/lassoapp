# Lasso Regression Analysis Streamlit App

This interactive application demonstrates the concepts of **Lasso Regression** (Least Absolute Shrinkage and Selection Operator) using the extended Boston Housing dataset. It allows users to explore how regularization strength affects model coefficients and predictive performance.

## Features

*   **Interactive Alpha Slider**: Adjust the regularization parameter ($\alpha$) from 0 to 2 to see real-time effects on the model.
*   **Coefficient Visualization**: View the magnitude of model coefficients and identify which predictors are selected (non-zero) or eliminated (zero) by the Lasso model.
*   **Regularization Path**: Visualize the trajectories of all coefficients as $\alpha$ changes, highlighting the shrinking effect of regularization.
*   **Performance Metrics**: Track the Model's Training and Test $R^2$ scores and the number of selected features dynamically.
*   **Selected Predictors Table**: A detailed list of the specific features that remain in the model at the current alpha level.

## Dataset

The app uses the **Extended Boston Housing dataset** via `mglearn`. This dataset includes the original 13 features of the Boston Housing data plus all possible interaction terms (polynomials of degree 2), resulting in **104 features**. This high-dimensional nature makes it an excellent candidate for demonstrating Lasso's feature selection capabilities.

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jgikas89/lassoapp.git
    cd lassoapp
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

4.  **Open in Browser:**
    The app should open automatically. If not, navigate to `http://localhost:8501`.

## Dependencies

*   `streamlit`
*   `scikit-learn`
*   `mglearn`
*   `matplotlib`
*   `pandas`
*   `numpy`

## Author

**jgikas89**

