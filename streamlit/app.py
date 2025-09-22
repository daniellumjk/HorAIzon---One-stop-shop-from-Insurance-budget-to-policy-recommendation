import streamlit as st
import pandas as pd
import pickle
import random
from sklearn.preprocessing import LabelEncoder

# --- Loading csv files for use --- #
hospital_premium = pd.read_csv("daniellumjk/horaizon---one-stop-shop-from-insurance-budget-to-policy-recommendation/main/streamlit/hospital_plan_premiums.csv")
dpi_premium = pd.read_csv("daniellumjk/horaizon---one-stop-shop-from-insurance-budget-to-policy-recommendation/main/streamlit/dpi_premium_rates.csv")

# --- File Paths --- #
budget_model_path = 'xgb_regressor_model.pkl'
dpi_model_path = 'xgboost_classifier_model.pkl'

# --- App Config --- #
st.set_page_config(page_title="Insurance Spending Predictor", layout="wide")

# --- Load the Model --- #
@st.cache_resource
def load_model(path):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
# def load_encoded_model(path):
#     try:
#         with open(path, 'rb') as f:
#             model = pickle.load(f)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None
    
budget_model = load_model(budget_model_path)
dpi_model = load_model(dpi_model_path)

# -------------------------------
# First Form: Budget Prediction
# -------------------------------
with st.container(border=True):
    st.title("HorAIzon")
    st.write("Get your budget, compare your plans and get a recommendation all in one!")

with st.form("input_form"):
    st.subheader("User Information")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 30)
        monthly_income = st.number_input("Monthly Income (Take Home)", min_value=0, max_value=10000, value=4000)
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        smoker = st.selectbox("Smoker", ["Yes", "No"])
        own_car = st.selectbox("Own a Car", ["Yes", "No"])
        kids = st.slider("Number of Kids", 0, 5, 0)
        elderly_parents = st.selectbox("Have elderly parents?", ["Yes", "No"])
    
    with col2:
        home_loan_remaining = st.number_input("Home Loan Remaining ($)", min_value=0, value=0)
        years_left_on_home_loan = st.slider("Years Left on Home Loan", 0, 30, 0)
        any_other_loan = st.selectbox("Any other loan?", ["Yes", "No"])
        monthly_expenses = st.number_input("Monthly Expenses ($)", min_value=0, value=1500)
        travel_freq_year = st.slider("Travel Frequency (per year)", 0, 12, 1)
        savings = st.number_input("Savings ($)", min_value=0, value=20000)

    submitted = st.form_submit_button("Predict Spending")

if submitted and budget_model:
    # --- Build Input DataFrame --- #
    input_data = {
        'age': age,
        'monthly_income': monthly_income,
        'home_loan_remaining': home_loan_remaining,
        'years_left_on_home_loan': years_left_on_home_loan,
        'monthly_expenses': monthly_expenses,
        'kids': kids,
        'travel_freq_year': travel_freq_year,
        'savings': savings,
        'gender_f': 1 if gender == 'Female' else 0,
        'gender_m': 1 if gender == 'Male' else 0,
        'smoker_0': 1 if smoker == 'No' else 0,
        'smoker_1': 1 if smoker == 'Yes' else 0,
        'marital_status_married': 1 if marital_status == 'Married' else 0,
        'marital_status_single': 1 if marital_status == 'Single' else 0,
        'own_car_0': 1 if own_car == 'No' else 0,
        'own_car_1': 1 if own_car == 'Yes' else 0,
        'elderly_parents_0': 1 if elderly_parents == 'No' else 0,
        'elderly_parents_1': 1 if elderly_parents == 'Yes' else 0,
        'any_other_loan_0': 1 if any_other_loan == 'No' else 0,
        'any_other_loan_1': 1 if any_other_loan == 'Yes' else 0
    }
    features_order = list(input_data.keys())
    input_df = pd.DataFrame([input_data], columns=features_order)

    prediction = budget_model.predict(input_df)[0]

    hosp_prem_series = hospital_premium.loc[hospital_premium['age'] == age, 'avg_hosp_plan']
    hosp_prem = hosp_prem_series.item() if not hosp_prem_series.empty else 0
    monthly_hosp_prem = hosp_prem / 12
    monthly_pa_prem = 300 / 12
    final_budget = prediction - monthly_hosp_prem - monthly_pa_prem

    # Save results in session_state
    st.session_state.prediction_done = True
    st.session_state.prediction = prediction
    st.session_state.monthly_hosp_prem = monthly_hosp_prem
    st.session_state.monthly_pa_prem = monthly_pa_prem
    st.session_state.final_budget = final_budget
    st.session_state.age = age
    st.session_state.gender = gender
    st.session_state.smoker = smoker
    st.session_state.monthly_income = monthly_income

# --- Always show results if available ---
if "prediction_done" in st.session_state and st.session_state.prediction_done:
    with st.container(border=True):
        st.subheader("Prediction Results")
        col_pred, col_prem, col_final = st.columns(3)
        with col_pred:           
            st.metric("Predicted Monthly Insurance Budget", f"${st.session_state.prediction:.2f}")
        with col_prem:
            st.metric("Estimated Hospital Plan (A Class Ward)", f"${st.session_state.monthly_hosp_prem:.2f}")
            st.metric("Estimated Personal Accident Plan", f"${st.session_state.monthly_pa_prem:.2f}")
        with col_final:
            st.metric("Final budget for Life Insurance", f"${st.session_state.final_budget:.2f}")

# -------------------------------
# Second Form: Policy Recommendation
# -------------------------------
if "prediction_done" in st.session_state and st.session_state.prediction_done and dpi_model:
    with st.form("input_form_2"):
        st.subheader("Preferences")
        provider = st.selectbox("Provider Preference", [
            "No Preference", "AIA Singapore", "Great Eastern Life", "Prudential Assurance Company Singapore (Pte) Limited", 
            "Income Insurance Limited", "Manulife (Singapore) Pte. Ltd.", "Singapore Life Ltd.", 
            "HSBC Life (Singapore) Pte. Ltd.", "FWD SINGAPORE PTE. LTD.", "Tokio Marine Life Insurance Singapore Ltd", 
            "Etiqa Insurance Pte. Ltd.", "China Taiping Insurance (Singapore) Pte. Ltd.", 
            "China Life Insurance (Singapore) Pte. Ltd.", "LIC (Singapore) Pte Ltd"])
        critical_illness = st.selectbox("Do you want Critical Illness Coverage", ["Yes", "No"])
        existing_insurance = st.selectbox("Do you have any Existing Insurance", ["Yes", "No"])
        existing_insurance_amount = st.selectbox("What is the Sum Assured of your Existing Insurance", [0, 100000, 200000, 300000, 400000, 500000])
        existing_premium = st.number_input("How much are you currently paying", min_value=0, max_value=1000, value=0)
        submitted_2 = st.form_submit_button("Recommend Policy")

    if submitted_2:
        # --- Reuse inputs from session_state ---
        age = st.session_state.age
        gender = st.session_state.gender
        smoker = st.session_state.smoker
        monthly_income = st.session_state.monthly_income
        budget = st.session_state.final_budget

        if smoker == 'Yes':
            smoker_2 = 1
        else:
            smoker_2 = 0

        if gender == 'Male':
            gender_2 = 'm'
        else:
            gender_2 = 'f'

        # --- Your original recommender logic ---
        age_until = 65
        coverage_term = age_until - age

        def ins_provider(input):
            if input == "AIA Singapore":
                return {'entered_sg': 1931,
                        'most_notable_5': 1,
                        'market_share_top': 1,
                        'local_companies': 0,
                        'provide_hospital_ins': 1}
            elif input == "Great Eastern Life":
                return {'entered_sg': 1908,
                        'most_notable_5': 1,
                        'market_share_top': 1,
                        'local_companies': 1,
                        'provide_hospital_ins': 1}
            elif input == "Prudential Assurance Company Singapore (Pte) Limited":
                return {'entered_sg': 1931,
                        'most_notable_5': 1,
                        'market_share_top': 1,
                        'local_companies': 0,
                        'provide_hospital_ins': 1}                    
            elif input == "Income Insurance Limited":
                return {'entered_sg': 1970,
                        'most_notable_5': 1,
                        'market_share_top': 1,
                        'local_companies': 1,
                        'provide_hospital_ins': 1} 
            elif input == "Manulife (Singapore) Pte. Ltd.":
                return {'entered_sg': 1899,
                        'most_notable_5': 1,
                        'market_share_top': 1,
                        'local_companies': 0,
                        'provide_hospital_ins': 0}
            elif input == "Singapore Life Ltd.":
                return {'entered_sg': 2022,
                        'most_notable_5': 0,
                        'market_share_top': 1,
                        'local_companies': 1,
                        'provide_hospital_ins': 1}             
            elif input == "HSBC Life (Singapore) Pte. Ltd.":
                return {'entered_sg': 2022,
                        'most_notable_5': 0,
                        'market_share_top': 1,
                        'local_companies': 0,
                        'provide_hospital_ins': 1} 
            elif input == "FWD SINGAPORE PTE. LTD.":
                return {'entered_sg': 2016,
                        'most_notable_5': 0,
                        'market_share_top': 0,
                        'local_companies': 0,
                        'provide_hospital_ins': 0} 
            elif input == "Tokio Marine Life Insurance Singapore Ltd":
                return {'entered_sg': 1923,
                        'most_notable_5': 0,
                        'market_share_top': 0,
                        'local_companies': 0,
                        'provide_hospital_ins': 0} 
            elif input == "Etiqa Insurance Pte. Ltd.":
                return {'entered_sg': 2007,
                        'most_notable_5': 0,
                        'market_share_top': 0,
                        'local_companies': 0,
                        'provide_hospital_ins': 0}                   
            elif input == "China Taiping Insurance (Singapore) Pte. Ltd.":
                return {'entered_sg': 2003,
                        'most_notable_5': 0,
                        'market_share_top': 0,
                        'local_companies': 0,
                        'provide_hospital_ins': 0} 
            elif input == "China Life Insurance (Singapore) Pte. Ltd.":
                return {'entered_sg': 2015,
                        'most_notable_5': 0,
                        'market_share_top': 0,
                        'local_companies': 0,
                        'provide_hospital_ins': 0} 
            elif input == "LIC (Singapore) Pte Ltd":
                return {'entered_sg': 2013,
                        'most_notable_5': 0,
                        'market_share_top': 0,
                        'local_companies': 0,
                        'provide_hospital_ins': 0} 
            else:
                return {'entered_sg': random.choice([2007,1923,2003,2016,2015,1931, 1908, 2013, 1970,2022,1899]),
                        'most_notable_5': random.choice([0,1]),
                        'market_share_top': random.choice([0,1]),
                        'local_companies': random.choice([0,1]),
                        'provide_hospital_ins': random.choice([0,1])} 
            
        # Benchmark amounts for coverage and balance premium
        benchmark_death_sa = (monthly_income * 12 * 9) - existing_insurance_amount
        benchmark_ci_sa = monthly_income * 12 * 4
        balance_budget = budget - existing_premium

        def death_sa(dsa):
            if dsa >= 400000:
                return 400000
            elif dsa >= 300000:
                return 400000
            elif dsa >= 200000:
                return 300000
            elif dsa >= 100000:
                return 200000
            else:
                return 100000

        def prem_amt(prem):
            if prem == 400000 and critical_illness == 'Yes':
                return 1800
            elif prem == 400000 and critical_illness == 'No':
                return 525
            elif prem == 300000 and critical_illness == 'Yes':
                return 1450
            elif prem == 300000 and critical_illness == 'No':
                return 425
            elif prem == 200000 and critical_illness == 'Yes':
                return 900
            elif prem == 200000 and critical_illness == 'No':
                return 325
            elif prem == 100000 and critical_illness == 'Yes':
                return 450
            else:
                return 175

        user_data = {
            'age': age,
            'age_until': 65,
            'coverage_term': coverage_term,
            'critical_illness': 1 if critical_illness == 'Yes' else 0,
            'smoker': smoker_2,
            'gender':gender_2,
            'type': 'term',
            'provide_ci': 1 if critical_illness == 'Yes' else random.choice([0,1]),
            'sum_assured': death_sa(benchmark_death_sa),
            'annual_premium': prem_amt(benchmark_death_sa)
        } #  1 if smoker == 'Yes' else 0 'm' if gender == 'Male' else 'f'

        rest_of_data = ins_provider(provider)
        final_data = user_data | rest_of_data

        features_order_2 = [
            'coverage_term', 'annual_premium', 'sum_assured', 'critical_illness', 'type', 'age_until',
            'gender', 'smoker', 'age', 'most_notable_5', 'market_share_top', 'local_companies', 'provide_ci',
            'provide_hospital_ins', 'entered_sg'
        ]
        input_df_2 = pd.DataFrame([final_data], columns=features_order_2)
        # Rebuild encoder from csv
        encoder = LabelEncoder()
        encoder.fit(dpi_premium["policy_name"])   # use the same column you used during training

        prediction_idx = dpi_model.predict(input_df_2)[0]
        policy_name = encoder.inverse_transform([prediction_idx])[0]

        # --- Initialize variables with default values before conditional blocks ---
        # This prevents NameError if the matching row or similar policies are empty
        provider_name = "N/A"
        premium_amount = 0
        sum_assured = 0
        
        most_ex_policy_name = "N/A"
        most_ex_provider = "N/A"
        most_ex_premium = 0
        
        q75_policy_name = "N/A"
        q75_provider = "N/A"
        q75_premium = 0
        
        q25_policy_name = "N/A"
        q25_provider = "N/A"
        q25_premium = 0
        
        cheapest_policy_name = "N/A"
        cheapest_provider = "N/A"
        cheapest_premium = 0


        # First set to match the exact policy name to provider and premium amount
        matching_row = dpi_premium[
                        (dpi_premium['policy_name'] == policy_name) & 
                        (dpi_premium['age'] == age) & 
                        (dpi_premium['sum_assured'] == death_sa(benchmark_death_sa)) & 
                        (dpi_premium['gender'] == gender_2) & 
                        (dpi_premium['smoker'] ==  smoker_2) & 
                        (dpi_premium['critical_illness'] == 1 if critical_illness == 'Yes' else 0)
                        ] # if gender == 'Male' else 'f' 1 if smoker == 'Yes' else 0
        # Check if a matching row was found
        if not matching_row.empty:
            # Get the value from the 'premium' column for the matching row
            # .iloc[0] is used to get the first row of the filtered DataFrame
            provider_name = matching_row['provider'].iloc[0]
            premium_amount = matching_row['annual_premium'].iloc[0] /12
            sum_assured = matching_row['sum_assured'].iloc[0]

        # Second set to match for other policy options based on the same profile
        similar_polices = dpi_premium[
                        (dpi_premium['age'] == age) & 
                        (dpi_premium['sum_assured'] == death_sa(benchmark_death_sa)) & 
                        (dpi_premium['gender'] == gender_2) & 
                        (dpi_premium['smoker'] == smoker_2) & 
                        (dpi_premium['critical_illness'] == 1 if critical_illness == 'Yes' else 0)
                        ]
        # Check if a df of similar policies was found
        if not similar_polices.empty:
            # Getting the most expensive in that df
            most_ex_series = similar_polices.loc[similar_polices['annual_premium'].idxmax()]
            most_ex_policy_name = most_ex_series['policy_name']
            most_ex_provider = most_ex_series['provider']
            most_ex_premium = most_ex_series['annual_premium'] /12
            # Getting the policy at the 75th percentile based on premium
            # df['annual_premium'].quantile(0.75) → gives the 75th percentile value.
            # (df['annual_premium'] - q75).abs() → difference of each row from that value.
            # .idxmin() → finds the index where that difference is smallest (closest).
            # df.loc[...] → selects that row.
            q75_closest_series = similar_polices.loc[(similar_polices['annual_premium'] - similar_polices['annual_premium'].quantile(0.75)).abs().idxmin()]
            q75_policy_name = q75_closest_series['policy_name']
            q75_provider = q75_closest_series['provider']
            q75_premium = q75_closest_series['annual_premium'] / 12
            # Getting the policy at the 25th percentile based on premium
            q25_closest_series = similar_polices.loc[(similar_polices['annual_premium'] - similar_polices['annual_premium'].quantile(0.25)).abs().idxmin()]
            q25_policy_name = q25_closest_series['policy_name']
            q25_provider = q25_closest_series['provider']
            q25_premium = q25_closest_series['annual_premium'] / 12
            # Getting the cheapest in that df
            cheapest_series = similar_polices.loc[similar_polices['annual_premium'].idxmin()]
            cheapest_policy_name = cheapest_series['policy_name']
            cheapest_provider = cheapest_series['provider']
            cheapest_premium = cheapest_series['annual_premium'] / 12

        with st.container(border=True):
            st.subheader("Your Ideal Coverage")
            col_death, col_ci, col_budget = st.columns(3)
            with col_death:        
                st.metric("Your Ideal Death Sum Assured:", f"${benchmark_death_sa:.2f}")
            with col_ci:
                st.metric("Your Ideal Critical Illness Sum Assured:", f"${benchmark_ci_sa:.2f}")
            with col_budget:
                st.metric("Your Balance Budget is:", f"${balance_budget:.2f}")

        with st.container(border=True):
            st.subheader("Your Bought Coverage")
            col_new_death, col_new_ci = st.columns(2)
            with col_new_death:
                st.metric("Your newly bought Death Coverage:", f"${sum_assured}")
            with col_new_ci:
                st.metric("Your newly bought Critical Illness Coverage:", f"${sum_assured}")

        with st.container(border=True):
                st.subheader("Recommended")
                st.metric(f"Policy Name:", f"{policy_name}")
                st.metric(f"Provider:", f"{provider_name}")
                st.metric(f"Premium per month:", f"${premium_amount:.2f}")

        with st.container(border=True):
            st.subheader("Other Options")
            output_df = pd.DataFrame(
                        {
                        "Provider": [cheapest_provider, q25_provider, q75_provider, most_ex_provider],
                        "Policy Name": [cheapest_policy_name, q25_policy_name, q75_policy_name, most_ex_policy_name],
                        "Premium Monthly": [f"${cheapest_premium:.2f}", f"${q25_premium:.2f}", f"${q75_premium:.2f}", f"${most_ex_premium:.2f}"],
                        }
                        )

            st.table(output_df)


