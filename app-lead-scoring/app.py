from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib
import os
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Flask App Initialization ---
app = Flask(__name__)

# --- In-Memory Database for Dashboard ---
prediction_history = []

# --- Model Feature Columns ---
EXPECTED_COLUMNS = [
    'Lead Origin', 'Lead Source', 'Do Not Email', 'Do Not Call', 'TotalVisits',
    'Total Time Spent on Website', 'Page Views Per Visit', 'Last Activity',
    'Country', 'Specialization', 'What is your current occupation', 'Search',
    'Newspaper Article', 'X Education Forums', 'Newspaper', 'Digital Advertisement',
    'Through Recommendations', 'Tags', 'Lead Quality', 'City',
    'A free copy of Mastering The Interview', 'Last Notable Activity'
]

# --- Global variables for model and preprocessor ---
model = None
preprocessor = None

# --- Load Model and Preprocessor with Error Handling ---
@lru_cache(maxsize=1)
def load_model_assets():
    """Load model assets with caching to prevent repeated loading"""
    global model, preprocessor
    try:
        if model is None or preprocessor is None:
            logger.info("Loading model and preprocessor...")
            model = joblib.load('log_reg_model.joblib')
            preprocessor = joblib.load('preprocessor.joblib')
            logger.info("Model and preprocessor loaded successfully.")
        return model, preprocessor
    except FileNotFoundError as e:
        logger.error(f"Model or preprocessor files not found: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading model assets: {e}")
        return None, None

# Initialize models on startup
model, preprocessor = load_model_assets()

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ezra H | Lead Scoring & Dashboard</title>
    <link rel="icon" type="image/x-icon" href="https://ezrahernowo.com/assets/imgs/favicon.png">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gray-50 text-gray-800">

    <div class="container mx-auto p-4 md:p-8">
        <header class="text-center mb-10">
            <h1 class="text-4xl md:text-5xl font-bold text-gray-800">Lead Scoring Pipeline</h1>
            <p class="text-lg text-gray-600 mt-2">Enter lead details to get a real-time conversion prediction.</p>
        </header>

        {% if error %}
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            <strong>Error:</strong> {{ error }}
        </div>
        {% endif %}

        {% if not model_loaded %}
        <div class="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded mb-4">
            <strong>Warning:</strong> Model not loaded. Please ensure model files are present.
        </div>
        {% endif %}

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            <!-- Input Form Section -->
            <div class="lg:col-span-1 bg-white p-6 rounded-2xl shadow-lg border border-gray-200">
                <h2 class="text-2xl font-semibold mb-6 border-b pb-4">New Lead Entry</h2>
                <form action="/" method="post">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        
                        <!-- Numerical Inputs -->
                        <div>
                            <label for="TotalVisits" class="block text-sm font-medium text-gray-700">Total Visits</label>
                            <input type="number" name="TotalVisits" value="{{ form_data.get('TotalVisits', '3.0') }}" 
                                   class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2">
                        </div>
                        <div>
                            <label for="Page Views Per Visit" class="block text-sm font-medium text-gray-700">Page Views Per Visit</label>
                            <input type="number" step="0.1" name="Page Views Per Visit" value="{{ form_data.get('Page Views Per Visit', '2.0') }}" 
                                   class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2">
                        </div>
                        <div class="md:col-span-2">
                             <label for="Total Time Spent on Website" class="block text-sm font-medium text-gray-700">Time on Website (s)</label>
                            <input type="number" name="Total Time Spent on Website" value="{{ form_data.get('Total Time Spent on Website', '500') }}" 
                                   class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2">
                        </div>

                        <!-- Categorical Inputs -->
                        <div>
                            <label for="Lead Origin" class="block text-sm font-medium text-gray-700">Lead Origin</label>
                            <select name="Lead Origin" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2">
                                <option {% if form_data.get('Lead Origin') == 'API' %}selected{% endif %}>API</option>
                                <option {% if form_data.get('Lead Origin') == 'Landing Page Submission' %}selected{% endif %}>Landing Page Submission</option>
                                <option {% if form_data.get('Lead Origin') == 'Lead Add Form' %}selected{% endif %}>Lead Add Form</option>
                                <option {% if form_data.get('Lead Origin') == 'Lead Import' %}selected{% endif %}>Lead Import</option>
                                <option {% if form_data.get('Lead Origin') == 'Quick Add Form' %}selected{% endif %}>Quick Add Form</option>
                            </select>
                        </div>
                         <div>
                            <label for="Lead Source" class="block text-sm font-medium text-gray-700">Lead Source</label>
                            <select name="Lead Source" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2">
                                <option {% if form_data.get('Lead Source') == 'Google' %}selected{% endif %}>Google</option>
                                <option {% if form_data.get('Lead Source') == 'Direct Traffic' %}selected{% endif %}>Direct Traffic</option>
                                <option {% if form_data.get('Lead Source') == 'Olark Chat' %}selected{% endif %}>Olark Chat</option>
                                <option {% if form_data.get('Lead Source') == 'Organic Search' %}selected{% endif %}>Organic Search</option>
                                <option {% if form_data.get('Lead Source') == 'Reference' %}selected{% endif %}>Reference</option>
                            </select>
                        </div>
                         <div class="md:col-span-2">
                            <label for="Last Activity" class="block text-sm font-medium text-gray-700">Last Activity</label>
                            <select name="Last Activity" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2">
                                <option {% if form_data.get('Last Activity') == 'Email Opened' %}selected{% endif %}>Email Opened</option>
                                <option {% if form_data.get('Last Activity') == 'SMS Sent' %}selected{% endif %}>SMS Sent</option>
                                <option {% if form_data.get('Last Activity') == 'Olark Chat Conversation' %}selected{% endif %}>Olark Chat Conversation</option>
                                <option {% if form_data.get('Last Activity') == 'Page Visited on Website' %}selected{% endif %}>Page Visited on Website</option>
                                <option {% if form_data.get('Last Activity') == 'Converted to Lead' %}selected{% endif %}>Converted to Lead</option>
                        </div>
                        <div class="md:col-span-2">
                            <label for="What is your current occupation" class="block text-sm font-medium text-gray-700">Current Occupation</label>
                            <select name="What is your current occupation" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2">
                                <option {% if form_data.get('What is your current occupation') == 'Unemployed' %}selected{% endif %}>Unemployed</option>
                                <option {% if form_data.get('What is your current occupation') == 'Working Professional' %}selected{% endif %}>Working Professional</option>
                                <option {% if form_data.get('What is your current occupation') == 'Student' %}selected{% endif %}>Student</option>
                                <option {% if form_data.get('What is your current occupation') == 'Housewife' %}selected{% endif %}>Housewife</option>
                                <option {% if form_data.get('What is your current occupation') == 'Other' %}selected{% endif %}>Other</option>
                        </div>

                        <!-- Hidden inputs -->
                        <input type="hidden" name="Do Not Email" value="No">
                        <input type="hidden" name="Do Not Call" value="No">
                        <input type="hidden" name="Country" value="India">
                        <input type="hidden" name="Specialization" value="Not Specified">
                        <input type="hidden" name="Search" value="No">
                        <input type="hidden" name="Newspaper Article" value="No">
                        <input type="hidden" name="X Education Forums" value="No">
                        <input type="hidden" name="Newspaper" value="No">
                        <input type="hidden" name="Digital Advertisement" value="No">
                        <input type="hidden" name="Through Recommendations" value="No">
                        <input type="hidden" name="Tags" value="Not Specified">
                        <input type="hidden" name="Lead Quality" value="Not Specified">
                        <input type="hidden" name="City" value="Mumbai">
                        <input type="hidden" name="A free copy of Mastering The Interview" value="No">
                        <input type="hidden" name="Last Notable Activity" value="Modified">
                    </div>
                    <button type="submit" class="mt-6 w-full bg-indigo-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-indigo-700 transition duration-300" 
                            {% if not model_loaded %}disabled{% endif %}>
                        {% if model_loaded %}Predict Score{% else %}Model Not Available{% endif %}
                    </button>
                </form>
            </div>

            <!-- Dashboard / Output Section -->
            <div class="lg:col-span-2 bg-white p-6 rounded-2xl shadow-lg border border-gray-200">
                <h2 class="text-2xl font-semibold mb-6 border-b pb-4">Dashboard</h2>
                
                <!-- Latest Prediction -->
                <div class="text-center bg-gray-50 p-6 rounded-xl mb-6">
                    <h3 class="text-lg font-medium text-gray-600 mb-2">Latest Prediction</h3>
                    {% if prediction is not none %}
                        <p class="text-3xl font-bold {{ 'text-green-600' if prediction == 1 else 'text-red-600' }}">
                            {{ 'Will Convert' if prediction == 1 else 'Will Not Convert' }}
                        </p>
                        <p class="text-xl text-gray-800 mt-1">Lead Score: <span class="font-bold">{{ "%.2f"|format(lead_score) }}%</span></p>
                    {% else %}
                        <p class="text-xl text-gray-500">Submit a lead to see the prediction.</p>
                    {% endif %}
                </div>

                <!-- History Table and Chart -->
                <div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
                    <div class="overflow-x-auto">
                        <h3 class="text-lg font-medium text-gray-600 mb-2">Prediction History</h3>
                        {% if history %}
                        <table class="min-w-full divide-y divide-gray-200">
                            <thead class="bg-gray-100">
                                <tr>
                                    <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Time on Site</th>
                                    <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Prediction</th>
                                    <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Score (%)</th>
                                </tr>
                            </thead>
                            <tbody class="bg-white divide-y divide-gray-200">
                                {% for item in history %}
                                    <tr>
                                        <td class="px-4 py-2 whitespace-nowrap text-sm">{{ item.inputs['Total Time Spent on Website'] }}s</td>
                                        <td class="px-4 py-2 whitespace-nowrap text-sm font-semibold {{ 'text-green-600' if item.prediction == 1 else 'text-red-600' }}">
                                            {{ 'Convert' if item.prediction == 1 else 'No Convert' }}
                                        </td>
                                        <td class="px-4 py-2 whitespace-nowrap text-sm font-bold">{{ "%.2f"|format(item.score) }}</td>
                                    </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                        {% else %}
                        <p class="text-gray-500">No predictions yet.</p>
                        {% endif %}
                    </div>
                    <div>
                        <h3 class="text-lg font-medium text-gray-600 mb-2">Conversion Summary</h3>
                         <div class="w-full h-64 flex items-center justify-center">
                            {% if history %}
                            <canvas id="conversionChart"></canvas>
                            {% else %}
                            <p class="text-gray-500">No data to display</p>
                            {% endif %}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    {% if history %}
    <script>
        const historyData = {{ history|tojson }};
        const convertedCount = historyData.filter(item => item.prediction == 1).length;
        const notConvertedCount = historyData.length - convertedCount;
        
        if (historyData.length > 0) {
            const ctx = document.getElementById('conversionChart').getContext('2d');
            new Chart(ctx, {
                type: 'doughnut',
                data: { 
                    labels: ['Will Convert', 'Will Not Convert'], 
                    datasets: [{ 
                        data: [convertedCount, notConvertedCount], 
                        backgroundColor: ['#10B981', '#EF4444'], 
                        borderColor: '#FFFFFF', 
                        borderWidth: 4 
                    }] 
                },
                options: { 
                    responsive: true, 
                    plugins: { 
                        legend: { position: 'bottom' }, 
                        tooltip: { 
                            callbacks: { 
                                label: function(c) { 
                                    let l = c.label || ''; 
                                    if(l){l+=': ';} 
                                    if(c.parsed!==null){
                                        const t = c.chart.data.datasets[0].data.reduce((a, b) => a + b, 0); 
                                        const p = t > 0 ? (c.raw/t*100).toFixed(1) + '%' : '0%'; 
                                        l += c.raw + ' (' + p + ')';
                                    } 
                                    return l; 
                                } 
                            } 
                        } 
                    } 
                }
            });
        }
    </script>
    {% endif %}
</body>
</html>
"""

def safe_predict(input_data):
    """Safely make predictions with error handling"""
    try:
        # Ensure models are loaded
        if model is None or preprocessor is None:
            logger.error("Model or preprocessor not loaded")
            return None, None, "Model not loaded"

        # Create DataFrame with expected columns
        data_for_df = {col: [input_data.get(col)] for col in EXPECTED_COLUMNS}
        input_df = pd.DataFrame(data_for_df)

        # Handle numerical columns
        numerical_cols = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
        for col in numerical_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
        
        # Preprocess and predict
        processed_input = preprocessor.transform(input_df)
        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0][1]
        lead_score = probability * 100

        return int(prediction), float(lead_score), None

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, None, f"Prediction failed: {str(e)}"

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    lead_score_result = None
    error_message = None
    form_data = {}
    
    if request.method == 'POST':
        form_data = request.form.to_dict()
        
        # Make prediction
        prediction_result, lead_score_result, error_message = safe_predict(form_data)
        
        # Store successful predictions
        if prediction_result is not None and error_message is None:
            prediction_history.insert(0, {
                "inputs": form_data,
                "prediction": prediction_result,
                "score": lead_score_result
            })
            # Keep history manageable
            if len(prediction_history) > 10:
                prediction_history.pop()

    # Check if model is loaded
    model_loaded = model is not None and preprocessor is not None

    return render_template_string(HTML_TEMPLATE, 
                                prediction=prediction_result, 
                                lead_score=lead_score_result, 
                                history=prediction_history,
                                error=error_message,
                                model_loaded=model_loaded,
                                form_data=form_data)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'predictions_count': len(prediction_history)
    }
    return status

# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Preprocessor loaded: {preprocessor is not None}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)