# Venv/script/activate
# streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="E-commerce Churn & LTV Predictor", layout="wide")

st.title("🛒 E-commerce Customer Churn & LTV Prediction ")
st.markdown("""
Upload an e-commerce dataset (CSV / XLSX / XLS) and choose whether you want to:
- Predict **Customer Churn**
- Predict **Customer Lifetime Value (LTV)**
- Predict **Both**
""")

# =========================
# HELPER FUNCTIONS
# =========================Venv

def load_uploaded_file(uploaded_file):
    """
    Load CSV, XLSX, or XLS file automatically
    """
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
        except:
            df = pd.read_csv(uploaded_file, encoding="latin1")
        return df

    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    else:
        raise ValueError("Unsupported file format. Please upload CSV, XLSX, or XLS.")


def detect_churn_column(df):
    """
    Try to auto-detect a churn column
    """
    possible_cols = [
        "churn", "is_churned", "churned", "retained",
        "status", "active", "is_active", "subscription_status"
    ]

    for col in df.columns:
        if str(col).lower().strip() in possible_cols:
            return col
    return None


def detect_ltv_column(df):
    """
    Try to auto-detect an LTV-like numeric column
    """
    possible_cols = [
        "ltv", "lifetime_value", "customer_lifetime_value",
        "clv", "revenue", "total_spent", "sales"
    ]

    for col in df.columns:
        if str(col).lower().strip() in possible_cols and pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def create_churn_target(df):
    """
    If churn column doesn't exist, help user create one.
    """
    st.subheader(" Churn Column  Found")

    st.markdown("""
Common churn columns: `churn`, `is_churned`, `retained`, `status`, `active`

You can create a churn target manually:
- Choose a column
- Choose the value that means customer churned
""")

    if df is None or df.empty:
        st.error("Dataset is empty or invalid.")
        return None, None

    col = st.selectbox("Select a column to create churn target", df.columns, key="manual_churn_col")

    valid_values = df[col].dropna().unique()
    if len(valid_values) == 0:
        st.error("Selected column has no usable values.")
        return None, None

    churn_value = st.selectbox("Select the value that means CHURNED", valid_values, key="manual_churn_value")

    new_df = df.copy()
    new_df["churn_target"] = (new_df[col] == churn_value).astype(int)

    counts = new_df["churn_target"].value_counts()

    st.write("### Churn Target Distribution")
    st.write(counts)

    if len(counts) < 2:
        st.error("Churn target has only one class. Please choose a different value/column.")
    elif counts.min() < 2:
        st.warning(
            "The app will train the chur prediction."
        )

    st.success("Created new churn target column: churn_target")
    return new_df, "churn_target"


def create_ltv_target(df):
    """
    If LTV column doesn't exist, allow user to choose a numeric target.
    """
    st.subheader(" LTV Column Found ")

    if df is None or df.empty:
        st.error("Dataset is empty or invalid.")
        return None, None


    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found in the dataset. Cannot create LTV target.")
        return None, None

    st.markdown("""
Select a numeric column that represents customer value, such as:
- total_spent
- revenue
- sales
- purchase_amount
- customer_value
""")

    ltv_col = st.selectbox("Select a numeric column for LTV prediction", numeric_cols, key="manual_ltv_col")

    if ltv_col is None:
        st.error("No valid numeric LTV column selected.")
        return None, None

    usable_values = pd.to_numeric(df[ltv_col], errors="coerce").dropna()
    if len(usable_values) == 0:
        st.error("Selected numeric column has no usable numeric values.")
        return None, None

    st.success(f"Selected LTV target column: {ltv_col}")
    return df.copy(), ltv_col


def preprocess_data(df, target_col, problem_type="classification"):
    """
    Preprocess dataset:
    - Remove rows with missing target
    - Drop feature columns with all null values
    - Fill missing feature values
    - Encode categorical features
    - Encode target if classification and target is categorical
    """
    working_df = df.copy()

    if target_col not in working_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Keep only rows where target exists
    working_df = working_df[working_df[target_col].notna()].copy()

    if len(working_df) < 5:
        raise ValueError("Not enough usable rows after removing missing target values.")

    X = working_df.drop(columns=[target_col]).copy()
    y = working_df[target_col].copy()

    if X.shape[1] == 0:
        raise ValueError("No feature columns available after removing target column.")

    # Drop columns that are fully empty (safe internal preprocessing only)
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError("All feature columns are empty after preprocessing.")

    # Fill missing feature values
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0
            X[col] = X[col].fillna(median_val)
        else:
            X[col] = X[col].fillna("Missing")

    encoders = {}

    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype) == "category" or X[col].dtype == "bool":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    # Handle target
    if problem_type == "classification":
        if y.dtype == "object" or str(y.dtype) == "category" or y.dtype == "bool":
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            encoders["target"] = target_encoder
        else:
            y = pd.to_numeric(y, errors="coerce")
            valid_idx = y.notna()
            X = X.loc[valid_idx].copy()
            y = y.loc[valid_idx].copy()
            working_df = working_df.loc[valid_idx].copy()

    else:
        # Regression target must be numeric
        y = pd.to_numeric(y, errors="coerce")
        valid_idx = y.notna()
        X = X.loc[valid_idx].copy()
        y = y.loc[valid_idx].copy()
        working_df = working_df.loc[valid_idx].copy()

    if len(X) < 5:
        raise ValueError("Not enough usable rows after preprocessing.")

    return X, y, working_df, encoders


# =========================
# MODEL FUNCTIONS
# =========================

def train_churn_model(df, target_col):
    """
    Train churn prediction model (classification)
    """
    X, y, processed_df, encoders = preprocess_data(df, target_col=target_col, problem_type="classification")

    class_counts = pd.Series(y).value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            f"Churn target column '{target_col}' has only one class. "
            f"Need at least 2 classes (e.g., 0 and 1) for classification."
        )

    use_stratify = class_counts.min() >= 2

    if not use_stratify:
        st.warning(" The app will train result on LTV Prediction.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None
    )

    # Safety check after split
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training split has only one class. Cannot train churn model properly.")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Probability scores for ROC
    y_proba = None
    roc_auc = None
    fpr, tpr = None, None

    if len(np.unique(y_test)) == 2:
        try:
            probs = model.predict_proba(X_test)
            if probs.shape[1] >= 2:
                y_proba = probs[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
        except:
            pass

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Predictions for all usable rows
    all_preds = model.predict(X)
    all_proba = None
    try:
        probs_all = model.predict_proba(X)
        if probs_all.shape[1] >= 2:
            all_proba = probs_all[:, 1]
    except:
        pass

    predictions_df = processed_df.copy()
    predictions_df["Predicted_Churn"] = all_preds
    if all_proba is not None:
        predictions_df["Churn_Probability"] = all_proba

    return {
        "model": model,
        "X": X,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "feature_names": X.columns,
        "processed_df": processed_df,
        "class_counts": class_counts,
        "used_stratify": use_stratify,
        "predictions_df": predictions_df
    }


def train_ltv_model(df, target_col):
    """
    Train LTV prediction model (regression)
    """
    X, y, processed_df, encoders = preprocess_data(df, target_col=target_col, problem_type="regression")

    if len(X) < 5:
        raise ValueError("Not enough rows to train LTV model. Please upload a larger dataset.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Predictions for all usable rows
    all_preds = model.predict(X)
    predictions_df = processed_df.copy()
    predictions_df["Predicted_LTV"] = all_preds

    return {
        "model": model,
        "X": X,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "feature_names": X.columns,
        "processed_df": processed_df,
        "predictions_df": predictions_df
    }


# =========================
# VISUALIZATION FUNCTIONS
# =========================

def plot_churn_confusion_matrix(cm):
    st.subheader("📊 Churn Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def plot_churn_roc_curve(result):
    if result["fpr"] is not None and result["tpr"] is not None and result["roc_auc"] is not None:
        st.subheader("📈 ROC Curve (Churn Model)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(result["fpr"], result["tpr"], label=f"AUC = {result['roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)
    else:
        st.info("ROC curve not available (requires binary classification with probability scores and 2 classes in test set).")


def plot_feature_importance(model, feature_names, title="Feature Importance"):
    st.subheader(f"⭐ {title}")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

    st.write("### Top Important Features")
    st.dataframe(feat_df.reset_index(drop=True), use_container_width=True)


def plot_ltv_actual_vs_predicted(y_test, y_pred):
    st.subheader("📉 LTV Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax.set_xlabel("Actual LTV")
    ax.set_ylabel("Predicted LTV")
    ax.set_title("Actual vs Predicted LTV")
    st.pyplot(fig)


def plot_ltv_residuals(y_test, y_pred):
    st.subheader("📊 LTV Residual Plot")
    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.7)
    ax.axhline(y=0, linestyle="--")
    ax.set_xlabel("Predicted LTV")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title("Residual Plot")
    st.pyplot(fig)


def download_csv(df, filename, label):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload your dataset file (CSV or Excel)",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        df = load_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    if df is None or df.empty:
        st.error("Uploaded dataset is empty or invalid.")
        st.stop()

    st.success("✅ File uploaded successfully!")

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.write("### Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    st.write("### Column Types")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]), use_container_width=True)

    # Prediction mode
    mode = st.radio(
        "Select Prediction Mode",
        ["Churn Only", "LTV Only", "Both Churn and LTV"]
    )

    working_df = df.copy()

    # =========================
    # CHURN ONLY
    # =========================
    if mode == "Churn Only":
        churn_col = detect_churn_column(working_df)

        if churn_col:
            st.success(f"Detected churn column automatically: {churn_col}")
            churn_target_col = churn_col
        else:
            working_df, churn_target_col = create_churn_target(working_df)
            if working_df is None or churn_target_col is None:
                st.stop()

        if st.button("Train Churn Model"):
            try:
                result = train_churn_model(working_df, churn_target_col)

                st.subheader("✅ Churn Model Results")
                st.metric("Accuracy", f"{result['accuracy']:.4f}")

                st.write("### Churn Class Distribution")
                class_df = result["class_counts"].reset_index()
                class_df.columns = ["Class", "Count"]
                st.dataframe(class_df, use_container_width=True)
                st.write(f"Used Stratified Split: {'Yes' if result['used_stratify'] else 'No'}")

                plot_churn_confusion_matrix(result["confusion_matrix"])
                plot_churn_roc_curve(result)
                plot_feature_importance(result["model"], result["feature_names"], title="Churn Feature Importance")

                st.subheader("📋 Churn Predictions")
                st.dataframe(result["predictions_df"].head(50), use_container_width=True)
                download_csv(result["predictions_df"], "churn_predictions.csv", "⬇️ Download Churn Predictions CSV")

            except Exception as e:
                st.error(f"Error training churn model: {e}")

    # =========================
    # LTV ONLY
    # =========================
    elif mode == "LTV Only":
        ltv_col = detect_ltv_column(working_df)

        if ltv_col:
            st.success(f"Detected LTV column automatically: {ltv_col}")
            ltv_target_col = ltv_col
        else:
            working_df, ltv_target_col = create_ltv_target(working_df)
            if working_df is None or ltv_target_col is None:
                st.stop()

        if st.button("Train LTV Model"):
            try:
                result = train_ltv_model(working_df, ltv_target_col)

                st.subheader("✅ LTV Model Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{result['mae']:.4f}")
                c2.metric("RMSE", f"{result['rmse']:.4f}")
                c3.metric("R² Score", f"{result['r2']:.4f}")

                plot_ltv_actual_vs_predicted(result["y_test"], result["y_pred"])
                plot_ltv_residuals(result["y_test"], result["y_pred"])
                plot_feature_importance(result["model"], result["feature_names"], title="LTV Feature Importance")

                st.subheader("📋 LTV Predictions")
                st.dataframe(result["predictions_df"].head(50), use_container_width=True)
                download_csv(result["predictions_df"], "ltv_predictions.csv", "⬇️ Download LTV Predictions CSV")

            except Exception as e:
                st.error(f"Error training LTV model: {e}")

    # =========================
    # BOTH CHURN + LTV
    # =========================
    elif mode == "Both Churn and LTV":
        # Churn
        churn_col = detect_churn_column(working_df)
        if churn_col:
            st.success(f"Detected churn column automatically: {churn_col}")
            churn_target_col = churn_col
        else:
            working_df, churn_target_col = create_churn_target(working_df)
            if working_df is None or churn_target_col is None:
                st.stop()

        # LTV
        ltv_col = detect_ltv_column(working_df)
        if ltv_col:
            st.success(f"Detected LTV column automatically: {ltv_col}")
            ltv_target_col = ltv_col
        else:
            working_df, ltv_target_col = create_ltv_target(working_df)
            if working_df is None or ltv_target_col is None:
                st.stop()

        if st.button("Train Both Models"):
            try:
                churn_result = train_churn_model(working_df, churn_target_col)
                ltv_result = train_ltv_model(working_df, ltv_target_col)

                st.subheader("✅ Both Models Trained Successfully")

                # Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Churn Accuracy", f"{churn_result['accuracy']:.4f}")
                with col2:
                    st.metric("LTV R² Score", f"{ltv_result['r2']:.4f}")

                st.write("### Churn Class Distribution")
                class_df = churn_result["class_counts"].reset_index()
                class_df.columns = ["Class", "Count"]
                st.dataframe(class_df, use_container_width=True)
                st.write(f"Used Stratified Split: {'Yes' if churn_result['used_stratify'] else 'No'}")

                st.markdown("---")
                st.header("📊 Churn Model Visualizations")
                plot_churn_confusion_matrix(churn_result["confusion_matrix"])
                plot_churn_roc_curve(churn_result)
                plot_feature_importance(churn_result["model"], churn_result["feature_names"], title="Churn Feature Importance")

                st.markdown("---")
                st.header("📈 LTV Model Visualizations")
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{ltv_result['mae']:.4f}")
                c2.metric("RMSE", f"{ltv_result['rmse']:.4f}")
                c3.metric("R² Score", f"{ltv_result['r2']:.4f}")

                plot_ltv_actual_vs_predicted(ltv_result["y_test"], ltv_result["y_pred"])
                plot_ltv_residuals(ltv_result["y_test"], ltv_result["y_pred"])
                plot_feature_importance(ltv_result["model"], ltv_result["feature_names"], title="LTV Feature Importance")

                # Separate prediction tables
                st.markdown("---")
                st.header("📋 Separate Prediction Outputs")

                st.subheader("Churn Predictions")
                st.dataframe(churn_result["predictions_df"].head(50), use_container_width=True)
                download_csv(churn_result["predictions_df"], "churn_predictions.csv", "⬇️ Download Churn Predictions CSV")

                st.subheader("LTV Predictions")
                st.dataframe(ltv_result["predictions_df"].head(50), use_container_width=True)
                download_csv(ltv_result["predictions_df"], "ltv_predictions.csv", "⬇️ Download LTV Predictions CSV")

                # Combined predictions
                st.markdown("---")
                st.header("📦 Combined Predictions")

                combined_df = churn_result["predictions_df"].copy()

                if "Predicted_LTV" in ltv_result["predictions_df"].columns:
                    if len(ltv_result["predictions_df"]) == len(combined_df):
                        combined_df["Predicted_LTV"] = ltv_result["predictions_df"]["Predicted_LTV"].values
                    else:
                        st.warning("⚠️ Could not combine LTV predictions due to row mismatch after preprocessing.")

                st.dataframe(combined_df.head(50), use_container_width=True)
                download_csv(combined_df, "combined_churn_ltv_predictions.csv", "⬇️ Download Combined Predictions CSV")

            except Exception as e:
                st.error(f"Error training both models: {e}")
else:
    st.info("📂 Please upload a CSV, XLSX, or XLS dataset to begin.")