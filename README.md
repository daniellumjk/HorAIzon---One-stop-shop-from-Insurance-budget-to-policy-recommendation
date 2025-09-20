# HorAIzon - A One Stop Shop for Direct Purchase Insurance to budget, compare and recommend
<img width="1334" height="752" alt="cover" src="https://github.com/user-attachments/assets/d5a5486c-5e4d-4d69-b5ed-379d1cae6c1d" />

<p style="color:#6a737d; font-style:italic;">
  Presented By: Daniel Lum 
</p>

## Content Directory
- Background
- Project Motivation
- How It Works
- Getting Started
- Sample EDA 
- Data Dictionaries
- Model Results
- Sample Output
- Conclusion
- App Deployment
- Additional Links
- Disclaimer

## Background
In Singapore, there is an alarming number of Singaporeans who are underinsured. This is due to several reasons: Having no idea where to start, lack of information on how much to set aside or how much they should be covered for and insurance not being a priority. Although there are benchmark figures to inform people how much they should be covered for or how much they should be spending on insurance, the information online is not convenient. They would have to hop from one website to another to calculate their budget, make comparisons and then decide for themselves which policy they would like to buy (Only for Direct Purchase Insurance (DPI)).

Alternatviely, they can speak to insurance agents to get the full service. However, that there is this "Friend-as-Agent" Dynamic that is prevalent in Singapore. What this is is friends or acquaintances who are insurance agents. And people often feel a sense of obligation or trust a familiar face when it comes to a complex product like insurance. However, this can also lead to issues if the agent is inexperienced, leaves the industry, or prioritizes sales targets over the client's needs. These agents are also restricted by what they can offer. Only offering products that are from the company they are with. Furthermore, compared to DPI policies, they are more costly. As at 2022, DPI products have been growing in popularity as well, making up 40.7% of the insurance market.

## Project Motivation
As mentioned earlier, if the general public wishes to pursue DPI products, they may not be well-equipped to serve their insurance needs. With the added inconvenience for information, We have decided to develop an AI tool where the general public can get an accurate budget for insurance spending, compare plans across 13 different companies and get a recommendation. Our product will be able to do the following:
- **Personalized Budgeting**: Utilises Machine Learning to calculate a personalized monthly insurance budget based on unique user data like income, expenses, and family status
- **Direct Purchase Insurance (DPI) Comparison**: Highlight the ability to compare different DPI policies in one place
- **Policy Recommendations**: Emphasize that the system recommends the "best" policy based on the user's budget and lifestyle

Our product brings all this information together in one app and it takes a data-driven approach to make recommendations rather than rely on benchmark figures. With everything in one place, our product is fast and reliable.

## How It Works 
This tool is built on 2 Machine Learning models. Firstly, a Regressor model which will predict a user's monthly insurance spending based on their inputs (age, income, and marital status, etc). Secondly, a Classifier model which will take the initial inputs plus a couple more to make the policy recommendation. 
- **Step 1**: The user inputs their personal data
- **Step 2**: The system uses the Regressor Model (XGBoost) to predict an appropriate monthly insurance budget
- **Step 3**: The system then uses the Classifier Model (XGBoost) to recommend the best DPI policy that fits their predicted budget and lifestyle

With the data provided, another 4 more policies providing the same coverage will also be shown, so that the user should they not choose to go with the recommendation, have other options to choose from.

## Getting Started
**The code blocks can be found in the 'code' folder**

The prerequisite libraries are:
```
python ==3.11.5
matplotlib==3.7.5
numpy==2.3.3
pandas==2.3.2
scikit_learn==1.4.2
seaborn==0.13.2
xgboost==3.0.5
```

## Sample EDA

1. New parents looking to start a new family should consider planning ahead for their life insurance and lock in their premiums before the arrival of their first child.
<img width="1193" height="713" alt="eda_1" src="https://github.com/user-attachments/assets/068a959d-2660-473f-bfd9-66215344ecb1" />

2. Smokers can now have a better view of which comapnies they can consider. 
<img width="1300" height="742" alt="eda_2" src="https://github.com/user-attachments/assets/948473d2-6766-45d1-acdd-d76ee459dc2e" />

## Data Dictionary: person_data
**This dataset can be found in the 'data' folder**

The following table details the data and their descriptions:
| Column Name	| Description	| Data Type |
| --- | --- | --- | 
| age | How old a person is | Integer |
| gender | Male or Female | String |
| monthly_income | The take home salary  | Integer |
| annual_income | Annualised take home salary | Integer |
| home_loan_remaining | The value of loan left to be repaid | Integer |
| years_left_on_home_loan | Number of years left to repay the home loan | Integer |
| monthly_expenses | The estimated expenses per month | Integer |
| smoker | Binary column, 1 for yes and 0 for no to whether the person is a smoker | Integer |
| marital_status | Married or Single. The legal relationship standing | String |
| kids | The number of children a person has | Integer |
| own_car | Binary column, 1 for yes and 0 for no to whether a person owns a car | Integer |
| elderly_parents | Binary column, 1 for yes and 0 for no to whether a person has elderly parents to care for | Integer |
| travel_freq_year | The estimated number of times a person travels for holiday  | Integer |
| any_other_loan | Binary column, 1 for yes and 0 for no to whether a person is servicing any other loan | Integer |
| savings | Estimated savings in bank account | Float |
| monthly_insurance_spending | The target column of this dataset. How much should a person spend on insurance | Float |

## Data Dictionary: dpi_premium_rates
**This dataset can be found in the 'data' folder**

The following table details the data and their descriptions:
| Column Name	| Description	| Data Type |
| --- | --- | --- | 
| policy_name | The target column of this dataset. The name and type of insurance product  | String |
| coverage_term | How long the policy would be run for | Integer |
| annual_premium | How much the policy costs per year  | Integer |
| annual_income | Annualised take home salary | Integer |
| provider | The name of the insuring company | String |
| sum_assured | The amount in the event of death or critical illness that will be paid out | Integer |
| critical_illness | Binary column, 1 for yes and 0 for no to whether the policy has critical illness coverage or not | Integer |
| type | The kind of insurance product | String |
| age_until | The age of the insured when the policy will expire | Integer |
| gender | Male or Female | String |
| smoker | Binary column, 1 for yes and 0 for no to whether the policy has accounted if the insured is a smoker | Integer |
| age | The entry age of the insured | Integer |
| most_notable_5 | Binary column, 1 for yes and 0 for no to whether the policy provider is either AIA, GE, Income, Manulife or Prudential | Integer |
| market_share_top | Binary column, 1 for yes and 0 for no to whether the insuring company is among the top few that holds majority market share | Integer |
| local_companies | Binary column, 1 for yes and 0 for no to whether the insuring company was founded in Singapore, GE, Income, Singlife | Integer |
| provide_ci | Binary column, 1 for yes and 0 for no to whether the insurer provides critical illness coverage | Integer |
| provide_hospital_ins | Binary column, 1 for yes and 0 for no to whether the insurer provides hospital insurace (Integrated shield plan) | Integer |
| entered_sg | The year with which the insurer was founded or entered Singapore | Integer |

## Results (Regressor Models)
Four machine learning models were built: XGBoost, LightGBM, Random Forest, Extra Trees.

The aim of these ML models was to learn from the dataset to predict how much someone should spend on insurance per month.

The following table summarizes the performance of the models:

<table style="border:1px solid #d0d7de; border-collapse:collapse; width:100%;">
  <thead>
    <tr>
      <th style="padding:6px 10px;">Model</th>
      <th style="padding:6px 10px;">Train Accuracy</th>
      <th style="padding:6px 10px;">Test Accuracy</th>
      <th style="padding:6px 10px;">RMSE</th>
      <th style="padding:6px 10px;">Runtime</th>
    </tr>
  </thead>
  <tbody>
<tr style="background:#fff3cd;"> <!-- Light yellow highlight -->
      <td><strong>XGBoost</strong></td>
      <td><strong>>99%<strong></td><td><strong>>99%<strong></td><td><strong>S$30.44<strong></td><td><strong>8ms<strong></td>
    </tr>
    </tr>
    <tr>
      <td style="padding:6px 10px;">LightGBM</td>
      <td style="padding:6px 10px;">>99%</td>
      <td style="padding:6px 10px;">98.48%</td>
      <td style="padding:6px 10px;">S$40.25</td>
      <td style="padding:6px 10px;">8ms</td>
    </tr>
    <tr>
      <td style="padding:6px 10px;">Random Forest</td>
      <td style="padding:6px 10px;">>99%</td>
      <td style="padding:6px 10px;">97.25%</td>
      <td style="padding:6px 10px;">S$54.18</td>
      <td style="padding:6px 10px;">37ms</td>
    </tr>
        <tr>
      <td style="padding:6px 10px;">Extra Trees</td>
      <td style="padding:6px 10px;">>99%</td>
      <td style="padding:6px 10px;">97.36%</td>
      <td style="padding:6px 10px;">SS$53.05</td>
      <td style="padding:6px 10px;">34ms</td>
    </tr>
  </tbody>
</table>

## Results (Classifier Models)
Four machine learning models were built: XGBoost, LightGBM, Random Forest, Extra Trees.

The aim of these ML models was to learn from the dataset to recommend the best DPI policy based on the already predicted budget and lifestyle of the person.

The following table summarizes the performance of the models:

<table style="border:1px solid #d0d7de; border-collapse:collapse; width:100%;">
  <thead>
    <tr>
      <th style="padding:6px 10px;">Model</th>
      <th style="padding:6px 10px;">Train Accuracy</th>
      <th style="padding:6px 10px;">Test Accuracy</th>
      <th style="padding:6px 10px;">Runtime</th>
    </tr>
  </thead>
  <tbody>
<tr style="background:#fff3cd;"> <!-- Light yellow highlight -->
      <td><strong>XGBoost</strong></td>
      <td><strong>>99%<strong></td><td><strong>>99%<strong></td><td><strong>94ms<strong></td>
    </tr>
    </tr>
    <tr>
      <td style="padding:6px 10px;">LightGBM</td>
      <td style="padding:6px 10px;">>99%</td>
      <td style="padding:6px 10px;">>99%</td>
      <td style="padding:6px 10px;">155ms</td>
    </tr>
    <tr>
      <td style="padding:6px 10px;">Random Forest</td>
      <td style="padding:6px 10px;">>99%</td>
      <td style="padding:6px 10px;">93.47%</td>
      <td style="padding:6px 10px;">98ms</td>
    </tr>
        <tr>
      <td style="padding:6px 10px;">Extra Trees</td>
      <td style="padding:6px 10px;">>99%</td>
      <td style="padding:6px 10px;">93.68%</td>
      <td style="padding:6px 10px;">117ms</td>
    </tr>
  </tbody>
</table>


By evaluating the trade-offs between train and test accuracy, model discrepancies, and runtime, XGBoost Regressor/Classifier has been identified as the best model for demonstrating superior performance with no significant overfitting.

## Sample Output
<img width="984" height="245" alt="budget_prediction" src="https://github.com/user-attachments/assets/fa266944-e8a4-457c-896b-b4ecc684e3cc" />

<img width="731" height="303" alt="recommended_policy" src="https://github.com/user-attachments/assets/ec5e299f-304b-45fb-b65a-983af443dcba" />

<img width="1728" height="220" alt="other_options" src="https://github.com/user-attachments/assets/ae3640ec-514d-4818-aeff-3d9592724190" />

## Conclusion
The results show that our Machine Learning models can make predictions on budget and recommendations for policies well beyond 99% accuracy. Coupled with the added convenience of having budget to policy recommendation all in one space, our product proves to be a quick and reliable solution. Users now do not have to go from one website to another to get the information they need.

## App Deployment


## Additional Links and Info

data: dpi_premium_rates - [Comparefirst](https://www.comparefirst.sg)
      person_data - synthetic data, generated to fit the local Singapore context through AI


Disclaimer: This project was done as a capstone for educational purposes only. It is in no way any form of financial advice.

