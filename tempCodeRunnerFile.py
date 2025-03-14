from flask import Flask, request, render_template
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model_path = r'P:\Arthi\models\random_forest_model.pkl'
scaler_path = r'P:\Arthi\models\scaler.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Dummy dataset for testing (replace with your actual test data)
X_test = np.array([
    [105, 32.5, 160, 35, 200],
    [75, 22.5, 90, 60, 100],
    [95, 28.0, 140, 45, 170],
    [80, 25.0, 100, 50, 120],
    [110, 35.0, 180, 30, 220]
])

y_test = np.array([1, 0, 1, 0, 1])  # Actual labels (1: Metabolic Syndrome, 0: No Metabolic Syndrome)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    waist_circ = float(request.form['waist_circ'])
    bmi = float(request.form['bmi'])
    blood_glucose = float(request.form['blood_glucose'])
    hdl = float(request.form['hdl'])
    triglycerides = float(request.form['triglycerides'])
    
    # Prepare the input data
    input_data = np.array([[waist_circ, bmi, blood_glucose, hdl, triglycerides]])
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_label = 'Yes (Metabolic Syndrome predicted)' if prediction[0] == 1 else 'No (No Metabolic Syndrome predicted)'
    
    # Calculate confusion matrix and accuracy
    y_pred = model.predict(scaler.transform(X_test))  # Predict on the test set
    cm = confusion_matrix(y_test, y_pred)  # Confusion matrix
    accuracy = accuracy_score(y_test, y_pred)  # Accuracy
    
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    # Save the plot to a BytesIO object and encode it in base64 to send to the template
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    
    # Return prediction, confusion matrix, and accuracy to the user
    return render_template('index.html', prediction=prediction_label, accuracy=accuracy, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
