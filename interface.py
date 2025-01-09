import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
from karim import (
    tree_regression,
    forest_regression,
    data_in_2d,
    perform_calarnas,
    perform_dbscan,
    new_instance
)

def main():
    st.title("Data Mining Application")

    with st.sidebar:
        selected = option_menu(
            "Menu",
            ["Upload Dataset", "Regression", "Clustering"],
            icons=["cloud-upload", "graph-up", "diagram-3"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Upload Dataset":
        st.header("Upload Dataset")
        uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview")
            st.dataframe(df)
            st.session_state["dataset"] = df

    elif selected == "Regression":
        if "dataset" not in st.session_state:
            st.warning("Please upload a dataset first.")
        else:
            df = st.session_state["dataset"]
            df_target = pd.read_csv('target.csv')
            st.header("Regression Analysis")

            target = st.selectbox("Select the target variable:", ["Summer", "Winter", "Spring", "Autumn"])
            dff = pd.read_csv(f'{target}.csv')
            max_depth = st.slider("Select max depth for Decision Trees:", min_value=1, max_value=20, value=5)
            num_trees = st.slider("Select number of trees for Random Forest:", min_value=3, max_value=100, value=10)
            model_choice = st.selectbox("Choose Regression Model:", ["Decision Tree", "Random Forest"])

            trained_model = None

            if st.button("Run Regression Models"):
                if model_choice == "Decision Tree":
                    dt_results, sickit_results, mine, sickit = tree_regression(dff, df_target, max_depth, target)
                    trained_model = mine
                    st.subheader("Decision Tree Results")
                    st.json(dt_results)
                    st.json(sickit_results)
                elif model_choice == "Random Forest":
                    dt_results, sickit_results, mine, sickit = forest_regression(dff, df_target, max_depth, target, num_trees)
                    trained_model = mine
                    st.subheader("Random Forest Results")
                    st.write("Our Metrics")
                    st.json(dt_results)
                    st.write("Sckit Metrics")
                    st.json(sickit_results)

                if trained_model:
                    st.success("Model trained successfully!")
                    st.session_state["trained_model"] = trained_model
                else:
                    st.error("Model training failed, please try again.")

            st.subheader("Predict New Data Instance")
            PSurf_Autumn = st.number_input("Enter PSurf:", value=0.0)
            Rainf_Autumn = st.number_input("Enter Rainf:", value=0.0)
            Snowf_Autumn = st.number_input("Enter Snowf:", value=0.0)
            Tair_Autumn = st.number_input("Enter Tair:", value=0.0)
            Wind_Autumn = st.number_input("Enter Wind:", value=0.0)
            sand_topsoil = st.number_input("Enter sand % topsoil:", value=0.0)
            silt_topsoil = st.number_input("Enter silt % topsoil:", value=0.0)
            clay_topsoil = st.number_input("Enter clay % topsoil:", value=0.0)
            pH_water_topsoil = st.number_input("Enter pH water topsoil:", value=0.0)
            OC_topsoil = st.number_input("Enter OC % topsoil:", value=0.0)
            OC_subsoil = st.number_input("Enter OC % subsoil:", value=0.0)
            N_topsoil = st.number_input("Enter N % topsoil:", value=0.0)
            N_subsoil = st.number_input("Enter N % subsoil:", value=0.0)
            CEC_topsoil = st.number_input("Enter CEC topsoil:", value=0.0)
            CaCO3_topsoil = st.number_input("Enter CaCO3 % topsoil:", value=0.0)
            CN_topsoil = st.number_input("Enter C/N topsoil:", value=0.0)

            season = target 
            trained_model = st.session_state.get("trained_model", None)
            if st.button("Predict"):
                if trained_model:  
                    prediction = new_instance(
                        PSurf_Autumn, Rainf_Autumn, Snowf_Autumn, Tair_Autumn,
                        Wind_Autumn, sand_topsoil, silt_topsoil, clay_topsoil,
                        pH_water_topsoil, OC_topsoil, OC_subsoil, N_topsoil,
                        N_subsoil, CEC_topsoil, CaCO3_topsoil, CN_topsoil,
                        trained_model, season
                    )
                    st.markdown(
                        """
                        <style>
                        .fancy-prediction {
                            font-size: 2.5rem;
                            font-weight: bold;
                            text-align: center;
                            padding: 10px;
                            color: #ffffff;
                            background: linear-gradient(90deg, rgba(0,36,36,1) 0%, rgba(0,128,128,1) 50%, rgba(0,212,212,1) 100%);
                            border-radius: 15px;
                            box-shadow: 0px 4px 15px rgba(0,128,128,0.5);
                            animation: pulse 2s infinite;
                        }

                        @keyframes pulse {
                            0% { transform: scale(1); }
                            50% { transform: scale(1.1); }
                            100% { transform: scale(1); }
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(f'<div class="fancy-prediction">Prediction: {prediction}</div>', unsafe_allow_html=True)

                else:
                    st.warning("Please run a regression model first.")

    elif selected == "Clustering":
        if "dataset" not in st.session_state:
            st.warning("Please upload a dataset first.")
        else:
            df = st.session_state["dataset"]
            st.header("Clustering Analysis")

            clustering_method = st.selectbox("Choose clustering algorithm:", ["CLARANS", "DBSCAN"])

            if clustering_method == "CLARANS":
                k = st.slider("Select number of clusters (k):", min_value=2, max_value=10, value=3)
                numlocal = st.slider("Select number of local searches:", min_value=1, max_value=10, value=2)
                maxneighbor = st.slider("Select max neighbor swaps:", min_value=1, max_value=10, value=2)

                if st.button("Run CLARANS"):
                    sil_score, pca_plot = perform_calarnas(df, k, numlocal, maxneighbor)
                    st.write(f"Silhouette Score: {sil_score}")
                    st.pyplot(pca_plot)

            elif clustering_method == "DBSCAN":
                eps = st.slider("Select eps:", min_value=0.1, max_value=10.0, value=0.5)
                min_samples = st.slider("Select min samples:", min_value=1, max_value=20, value=5)

                if st.button("Run DBSCAN"):
                    sil_score, pca_plot = perform_dbscan(df, eps, min_samples)

                    if sil_score is not None:
                        st.write(f"Silhouette Score: {sil_score:.4f}")
                    else:
                        st.write("Error, 1 cluster occured.")

                    st.pyplot(pca_plot)


if __name__ == "__main__":
    main()
