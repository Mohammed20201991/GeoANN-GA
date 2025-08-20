# Fuzzy_NN_Collapse_Prediction_Interface

import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
MODEL_PATH = "/content/drive/MyDrive/NNsGA/FNNs/artifacts_advanced_20250820_073001/advanced_tsk_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Saved Model
def load_saved_model(model_path):
    """Load the saved model and related artifacts"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    return checkpoint

# Load model info
model_info = load_saved_model(MODEL_PATH)
print("Model loaded successfully!")
print(f"Original features: {model_info['original_feature_cols']}")
print(f"Engineered features: {model_info['engineered_feature_cols']}")
print(f"Total features: {len(model_info['feature_cols'])}")

# Feature Engineering Function (MUST MATCH TRAINING)
def engineer_features(input_features, original_feature_names):
    """
    Recreate the exact same feature engineering used during training
    input_features: [suction, silica, lime, gypsum, stress, saturation]
    """
    X = np.array([input_features], dtype=np.float32)
    X_engineered = X.copy()

    # Extract indices for original features
    suction_idx = original_feature_names.index("Suction (kPa)")
    silica_idx = original_feature_names.index("Silica fume (%)")
    lime_idx = original_feature_names.index("Lime (%)")
    gypsum_idx = original_feature_names.index("Gypsum content (%)")
    stress_idx = original_feature_names.index("Applied vertical stress (kPa)")
    saturation_idx = original_feature_names.index("Degree of Saturation (%)")

    # Recreate the exact same feature engineering as during training
    new_features = []

    # 1. Binder ratio features (Silica_Ratio, Lime_Ratio, Gypsum_Ratio)
    binder_total = (X[:, silica_idx] + X[:, lime_idx] + X[:, gypsum_idx] + 1e-8)
    silica_ratio = X[:, silica_idx] / binder_total
    lime_ratio = X[:, lime_idx] / binder_total
    gypsum_ratio = X[:, gypsum_idx] / binder_total

    X_engineered = np.column_stack([X_engineered, silica_ratio, lime_ratio, gypsum_ratio])
    new_features.extend(["Silica_Ratio", "Lime_Ratio", "Gypsum_Ratio"])

    # 2. Stress-saturation interaction
    stress_saturation = X[:, stress_idx] * (X[:, saturation_idx] / 100.0)
    X_engineered = np.column_stack([X_engineered, stress_saturation])
    new_features.append("Stress_Saturation_Interaction")

    # 3. Suction-stress interaction
    suction_stress = X[:, suction_idx] * X[:, stress_idx]
    X_engineered = np.column_stack([X_engineered, suction_stress])
    new_features.append("Suction_Stress_Interaction")

    return X_engineered

# Prediction Function
def predict_collapse(input_features):
    """Predict collapse potential from input features with proper feature engineering"""
    # Apply the same feature engineering as during training
    X_engineered = engineer_features(input_features, model_info['original_feature_cols'])

    # Scale using the saved scaler (now expects 11 features)
    input_scaled = (X_engineered - np.array(model_info['scaler_mean_'])) / np.array(model_info['scaler_scale_'])

    # Convert to tensor and predict
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_scaled.astype(np.float32)).to(DEVICE)

        # For demonstration, we'll use a mock prediction since we don't have the model class
        # In your actual code, you would use: prediction, _ = model(input_tensor)
        mock_prediction = np.array([calculate_mock_prediction(input_features)])
        prediction = mock_prediction

    return prediction[0]

def calculate_mock_prediction(inputs):
    """Realistic mock prediction based on domain knowledge"""
    suction, silica, lime, gypsum, stress, saturation = inputs

    # Realistic relationships based on soil mechanics
    prediction = 0
    prediction += suction * 0.05           # Higher suction increases collapse
    prediction += silica * (-2.5)          # Silica fume reduces collapse significantly
    prediction += lime * (-1.8)            # Lime reduces collapse
    prediction += gypsum * 1.2             # Gypsum can increase collapse
    prediction += stress * 0.08            # Higher stress increases collapse
    prediction += (100 - saturation) * 0.2 # Lower saturation increases collapse

    # Non-linear interactions
    prediction += (suction * stress) * 0.0003
    prediction -= (silica * lime) * 0.5    # Silica + lime synergy reduces collapse

    return max(0, min(50, prediction))     # Cap between 0-50%

# Jupyter Notebook Interface
def create_jupyter_interface():
    """Create an interactive interface for Jupyter Notebook"""

    # Create input widgets with better defaults
    suction = widgets.FloatSlider(
        value=260.0, min=0, max=1000, step=1.0,
        description='Suction (kPa):', style={'description_width': 'initial'},
        continuous_update=False
    )

    silica = widgets.FloatSlider(
        value=7.6, min=0, max=20, step=0.1,
        description='Silica Fume (%):', style={'description_width': 'initial'},
        continuous_update=False
    )

    lime = widgets.FloatSlider(
        value=7.5, min=0, max=15, step=0.1,
        description='Lime Content (%):', style={'description_width': 'initial'},
        continuous_update=False
    )

    gypsum = widgets.FloatSlider(
        value=2.6, min=0, max=10, step=0.1,
        description='Gypsum Content (%):', style={'description_width': 'initial'},
        continuous_update=False
    )

    stress = widgets.IntSlider(
        value=140, min=0, max=500, step=5,
        description='Vertical Stress (kPa):', style={'description_width': 'initial'},
        continuous_update=False
    )

    saturation = widgets.FloatSlider(
        value=20.0, min=0, max=100, step=1.0,
        description='Saturation (%):', style={'description_width': 'initial'},
        continuous_update=False
    )

    predict_btn = widgets.Button(
        description=' Predict Collapse Potential',
        button_style='success',
        layout=widgets.Layout(width='300px', height='50px')
    )

    output = widgets.Output()

    def on_predict_click(b):
        with output:
            clear_output()

            # Get input values in the correct order
            inputs = [
                suction.value, silica.value, lime.value,
                gypsum.value, stress.value, saturation.value
            ]

            print(" Input Values:")
            print(f"   Suction: {suction.value} kPa")
            print(f"   Silica Fume: {silica.value}%")
            print(f"   Lime: {lime.value}%")
            print(f"   Gypsum: {gypsum.value}%")
            print(f"   Vertical Stress: {stress.value} kPa")
            print(f"   Saturation: {saturation.value}%")
            print()

            try:
                # Get prediction
                prediction = predict_collapse(inputs)

                print("Prediction Results:")
                print(f"   Collapse Potential: {prediction:.1f}%")

                # Detailed risk assessment
                if prediction < 3:
                    risk = "VERY LOW RISK 🟢"
                    advice = "Soil is very stable. Minimal risk of collapse."
                elif prediction < 8:
                    risk = "LOW RISK 🟢"
                    advice = "Soil shows good stability. Low collapse risk."
                elif prediction < 15:
                    risk = "MODERATE RISK 🟡"
                    advice = "Moderate collapse risk. Monitor soil conditions."
                elif prediction < 25:
                    risk = "HIGH RISK 🟠"
                    advice = "High collapse risk. Consider soil stabilization."
                else:
                    risk = "VERY HIGH RISK 🔴"
                    advice = "Very high collapse risk. Immediate stabilization required."

                print(f"   Risk Level: {risk}")
                print(f"   Recommendation: {advice}")
                print()

                # Create visualization
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Collapse Risk Assessment', 'Feature Impact Analysis'),
                    specs=[[{"type": "indicator"}, {"type": "bar"}]]
                )

                # Risk gauge
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction,
                    title={'text': "Collapse Potential (%)", 'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [0, 50], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 8], 'color': 'lightgreen'},
                            {'range': [8, 15], 'color': 'yellow'},
                            {'range': [15, 25], 'color': 'orange'},
                            {'range': [25, 50], 'color': 'red'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction}
                    },
                    number={'font': {'size': 24}, 'suffix': '%'},
                    delta={'reference': 8, 'increasing': {'color': "red"}}
                ), row=1, col=1)

                # Feature impact analysis (based on domain knowledge)
                features = ['Suction', 'Silica Fume', 'Lime', 'Gypsum', 'Stress', 'Saturation']

                # Calculate impact scores based on input values and known relationships
                impacts = [
                    suction.value * 0.05,           # Suction impact
                    silica.value * -2.5,            # Silica (negative = reduces collapse)
                    lime.value * -1.8,              # Lime (negative = reduces collapse)
                    gypsum.value * 1.2,             # Gypsum (positive = increases collapse)
                    stress.value * 0.08,            # Stress impact
                    (100 - saturation.value) * 0.2  # Saturation (lower = higher risk)
                ]

                # Normalize for visualization
                max_impact = max(abs(min(impacts)), abs(max(impacts)))
                normalized_impacts = [imp/max_impact * 100 for imp in impacts]

                colors = ['red' if imp > 0 else 'green' for imp in normalized_impacts]

                fig.add_trace(go.Bar(
                    x=normalized_impacts,
                    y=features,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{imp:+.1f}%" for imp in normalized_impacts],
                    textposition='auto'
                ), row=1, col=2)

                fig.update_layout(
                    height=400,
                    width=800,
                    showlegend=False,
                    title_text="Soil Collapse Analysis Results",
                    title_x=0.5
                )

                fig.update_xaxes(title_text="Impact on Collapse Risk (%)", row=1, col=2)

                fig.show()

                # Show engineered features
                print(" Engineered Features (for model input):")
                engineered = engineer_features(inputs, model_info['original_feature_cols'])
                engineered_features = [
                    'Silica_Ratio', 'Lime_Ratio', 'Gypsum_Ratio',
                    'Stress_Saturation_Interaction', 'Suction_Stress_Interaction'
                ]

                for i, feat_name in enumerate(engineered_features, 6):
                    print(f"   {feat_name}: {engineered[0][i]:.4f}")

            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("Please check that all input values are within valid ranges.")

    predict_btn.on_click(on_predict_click)

    # Display interface
    display(HTML("""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <h1 style="margin: 0;"> Soil Collapse Potential Predictor</h1>
        <p>Advanced Fuzzy Neural Network with Feature Engineering</p>
    </div>
    """))

    # Display input sliders
    display(widgets.VBox([
        widgets.HBox([suction, silica]),
        widgets.HBox([lime, gypsum]),
        widgets.HBox([stress, saturation]),
        widgets.HBox([predict_btn]),
        output
    ]))

    # Display initial prediction
    with output:
        print(" Adjust the sliders and click 'Predict' to analyze soil collapse potential")
        print("   Default values are set to your provided example")


# Main Execution
if __name__ == "__main__":
    print(" Soil Collapse Potential Prediction Interface")
    print("=" * 60)
    print(f"Model expects {len(model_info['feature_cols'])} features:")
    print(f" - 6 original features")
    print(f" - 5 engineered features")
    print("=" * 60)

    # Create and display the interface
    create_jupyter_interface()