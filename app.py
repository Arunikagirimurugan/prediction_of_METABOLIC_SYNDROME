from flask import Flask, request, render_template
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Paths for the model and scaler files
MODEL_PATH = r'C:/Users/aruni/OneDrive/Desktop/METABOLIC_SYNDROME-main/METABOLIC_SYNDROME-main/models/random_forest_model.pkl'
SCALER_PATH = r'C:/Users/aruni/OneDrive/Desktop/METABOLIC_SYNDROME-main/METABOLIC_SYNDROME-main/models/scaler.pkl'

# Load the trained model and scaler
def load_pickle_file(file_path, file_type):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        raise RuntimeError(f"Error loading {file_type}: {e}")

model = load_pickle_file(MODEL_PATH, "model")
scaler = load_pickle_file(SCALER_PATH, "scaler")

# Test data (for evaluation, not for every prediction)
X_test = np.array([
    [105, 32.5, 160, 35, 200],
    [75, 22.5, 90, 60, 100],
    [95, 28.0, 140, 45, 170],
    [80, 25.0, 100, 50, 120],
    [110, 35.0, 180, 30, 220]
])
y_test = np.array([1, 0, 1, 0, 1])  # Actual labels

# Medical advice based on prediction
def get_medical_advice(condition):
    if condition == 'Metabolic Syndrome':
        return (
            "- Follow a low-carbohydrate diet to control blood sugar levels.\n"
            "- Include high-fiber foods like vegetables, fruits, and whole grains.\n"
            "- Exercise regularly for at least 30 minutes a day.\n"
            "- Medications like Metformin and statins may be prescribed to manage insulin resistance and cholesterol levels.\n"
            "- Monitor blood pressure and cholesterol levels regularly.\n"
        )
    return (
        "- Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.\n"
        "- Continue regular physical activity to maintain a healthy weight.\n"
        "- Stay hydrated and get sufficient sleep.\n"
        "- Monitor blood pressure, cholesterol, and blood glucose levels regularly.\n"
        "- Consider regular health check-ups to ensure overall well-being.\n"
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract user inputs
        waist_circ = request.form.get('waist_circ')
        bmi = request.form.get('bmi')
        blood_glucose = request.form.get('blood_glucose')
        hdl = request.form.get('hdl')
        triglycerides = request.form.get('triglycerides')
        
        # Validate input data
        if not all([waist_circ, bmi, blood_glucose, hdl, triglycerides]):
            return render_template('index.html', error="All fields are required!")
        
        try:
            waist_circ = float(waist_circ)
            bmi = float(bmi)
            blood_glucose = float(blood_glucose)
            hdl = float(hdl)
            triglycerides = float(triglycerides)
        except ValueError:
            return render_template('index.html', error="Invalid input data. Please enter numeric values.")
        
        # Prepare the input data
        input_data = np.array([[waist_circ, bmi, blood_glucose, hdl, triglycerides]])
        input_data_scaled = scaler.transform(input_data)
        
        # Make predictions
        prediction = model.predict(input_data_scaled)
        prediction_label = (
            'Yes (Metabolic Syndrome predicted)' 
            if prediction[0] == 1 else 
            'No (No Metabolic Syndrome predicted)'
        )
        
        # Provide medical advice
        advice = get_medical_advice('Metabolic Syndrome' if prediction[0] == 1 else 'No Metabolic Syndrome')
        
        # Evaluate model performance on test data (Optional - can be moved to another route for diagnostics)
        y_pred = model.predict(scaler.transform(X_test))
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close(fig)
        
        # Render results
        return render_template(
            'index.html', 
            prediction=prediction_label, 
            advice=advice, 
            accuracy=accuracy,  # Pass the raw accuracy value, no formatting needed
            plot_url=plot_url
        )
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {e}")

if __name__ == '_main_':
    app.run(debug=True)