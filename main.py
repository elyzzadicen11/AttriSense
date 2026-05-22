from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)

CORS(app)

# Load trained model
model = joblib.load('employee_attrition.pkl')


@app.route('/')
def home():

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.get_json()

        input_data = {

            # Numerical Inputs
            'Age': int(data['Age']),
            'MonthlyIncome': int(data['MonthlyIncome']),
            'DistanceFromHome': int(data['DistanceFromHome']),
            'JobSatisfaction': int(data['JobSatisfaction']),
            'WorkLifeBalance': int(data['WorkLifeBalance']),
            'EnvironmentSatisfaction': int(data['EnvironmentSatisfaction']),
            'YearsAtCompany': int(data['YearsAtCompany']),
            'StockOptionLevel': int(data['StockOptionLevel']),
            'NumCompaniesWorked': int(data['NumCompaniesWorked']),

            # Default Numerical Features
            'DailyRate': 0,
            'Education': 0,
            'HourlyRate': 0,
            'JobInvolvement': 0,
            'JobLevel': 0,
            'MonthlyRate': 0,
            'PercentSalaryHike': 0,
            'PerformanceRating': 0,
            'RelationshipSatisfaction': 0,
            'TotalWorkingYears': 0,
            'TrainingTimesLastYear': 0,
            'YearsInCurrentRole': 0,
            'YearsSinceLastPromotion': 0,
            'YearsWithCurrManager': 0,

            # One-hot Encoded Features
            'BusinessTravel_Travel_Frequently': 0,
            'BusinessTravel_Travel_Rarely': 0,

            'Department_Research & Development': 0,
            'Department_Sales': 0,

            'EducationField_Life Sciences': 0,
            'EducationField_Marketing': 0,
            'EducationField_Medical': 0,
            'EducationField_Other': 0,
            'EducationField_Technical Degree': 0,

            'Gender_Male': 0,

            'JobRole_Human Resources': 0,
            'JobRole_Laboratory Technician': 0,
            'JobRole_Manager': 0,
            'JobRole_Manufacturing Director': 0,
            'JobRole_Research Director': 0,
            'JobRole_Research Scientist': 0,
            'JobRole_Sales Executive': 0,
            'JobRole_Sales Representative': 0,

            'MaritalStatus_Married': 0,
            'MaritalStatus_Single': 0,

            'OverTime_Yes': 0
        }

        # Business Travel
        if data['BusinessTravel'] == 'Travel_Frequently':

            input_data[
                'BusinessTravel_Travel_Frequently'
            ] = 1

        elif data['BusinessTravel'] == 'Travel_Rarely':

            input_data[
                'BusinessTravel_Travel_Rarely'
            ] = 1

        # Overtime
        if data['OverTime'] == 'Yes':

            input_data['OverTime_Yes'] = 1

        # Job Role
        role_column = f"JobRole_{data['JobRole']}"

        if role_column in input_data:

            input_data[role_column] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Arrange columns in exact order
        model_columns = [
            'Age',
            'DailyRate',
            'DistanceFromHome',
            'Education',
            'EnvironmentSatisfaction',
            'HourlyRate',
            'JobInvolvement',
            'JobLevel',
            'JobSatisfaction',
            'MonthlyIncome',
            'MonthlyRate',
            'NumCompaniesWorked',
            'PercentSalaryHike',
            'PerformanceRating',
            'RelationshipSatisfaction',
            'StockOptionLevel',
            'TotalWorkingYears',
            'TrainingTimesLastYear',
            'WorkLifeBalance',
            'YearsAtCompany',
            'YearsInCurrentRole',
            'YearsSinceLastPromotion',
            'YearsWithCurrManager',
            'BusinessTravel_Travel_Frequently',
            'BusinessTravel_Travel_Rarely',
            'Department_Research & Development',
            'Department_Sales',
            'EducationField_Life Sciences',
            'EducationField_Marketing',
            'EducationField_Medical',
            'EducationField_Other',
            'EducationField_Technical Degree',
            'Gender_Male',
            'JobRole_Human Resources',
            'JobRole_Laboratory Technician',
            'JobRole_Manager',
            'JobRole_Manufacturing Director',
            'JobRole_Research Director',
            'JobRole_Research Scientist',
            'JobRole_Sales Executive',
            'JobRole_Sales Representative',
            'MaritalStatus_Married',
            'MaritalStatus_Single',
            'OverTime_Yes'
        ]

        input_df = input_df.reindex(
            columns=model_columns,
            fill_value=0
        )

        # Predict
        prediction_proba = model.predict_proba(
            input_df
        )[:, 1][0]

        prediction_label = (
            "Will Leave"
            if prediction_proba > 0.5
            else "Will Stay"
        )

        return jsonify({
            'prediction': prediction_label,
            'probability': float(prediction_proba)
        })

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 400


if __name__ == '__main__':

    app.run(debug=True, port=2725)
