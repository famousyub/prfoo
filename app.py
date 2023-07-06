# app.py
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from flask_cors  import CORS


app = Flask(__name__)

CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})
@app.route('/predict', methods=['POST'])
def predict_taux_retard():
    try:
        # Load the data from the CSV file
        data = pd.read_csv('./content/newdata.csv', delimiter=';')

        # Preprocess the data
        data['Date_Debut_prev'] = pd.to_datetime(data['Date_Debut_prev'], format='%d/%m/%Y')
        data['Date_fin_prev'] = pd.to_datetime(data['Date_fin_prev'], format='%d/%m/%Y')
        data['Date_Debut_reelle'] = pd.to_datetime(data['Date_Debut_reelle'], format='%d/%m/%Y')
        data['Date_fin_reelle'] = pd.to_datetime(data['Date_fin_reelle'], format='%d/%m/%Y')
        data['tauxRetard'] = data['tauxRetard'].str.replace(',', '.').astype(float)

        # Drop irrelevant columns (e.g., IDs)
        data = data.drop(columns=['ID_ministere', 'ID_mission', 'ID_missionnaire'])

        # Separate features and labels
        X = data.drop(columns=['tauxRetard'])
        y = data['tauxRetard']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model (using linear regression as an example)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Get the input data from the request
        input_data = request.get_json()

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data, index=[0])

        # Make predictions using the trained model
        predictions = model.predict(input_df)

        return jsonify({'prediction': predictions[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5002)
