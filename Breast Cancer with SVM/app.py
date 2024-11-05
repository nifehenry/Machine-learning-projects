import gradio as gd
import pickle
import numpy as np

# Load the breast cancer prediction model
with open("C:/Users/USER/Desktop/projects/Breast Cancer with SVM/model/Brest_cancer_prediction.pkl", "rb") as file:
    model = pickle.load(file)

# Load the scaler
with open("C:/Users/USER/Desktop/projects/Breast Cancer with SVM/Scaler/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Define a prediction function
def predict(concave_points_mean, symmetry_mean, radius_se, area_se,
             fractal_dimension_se, radius_worst, texture_worst, area_worst,
             concavity_worst, concave_points_worst):
    # Prepare input data for prediction
    input_data = np.array([concave_points_mean, symmetry_mean, radius_se, area_se,
                           fractal_dimension_se, radius_worst, texture_worst, area_worst,
                           concavity_worst, concave_points_worst]).reshape(1, -1)
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    # Predict using the loaded model
    prediction = model.predict(scaled_data)
    return "Malignant" if prediction[0] == 2 else "Benign"

# Define Gradio interface
iface = gd.Interface(
    fn=predict,
    inputs=[
        gd.components.Slider(label="Concave Points Mean", minimum=0, maximum=30, step=0.01),
        gd.components.Slider(label="Symmetry Mean", minimum=0, maximum=30, step=0.01),
        gd.components.Slider(label="Radius SE", minimum=0, maximum=30, step=0.01),
        gd.components.Slider(label="Area SE", minimum=0, maximum=3000, step=0.01),
        gd.components.Slider(label="Fractal Dimension SE", minimum=0, maximum=1, step=0.01),
        gd.components.Slider(label="Radius Worst", minimum=0, maximum=30, step=0.01),
        gd.components.Slider(label="Texture Worst", minimum=0, maximum=50, step=0.01),
        gd.components.Slider(label="Area Worst", minimum=0, maximum=3000, step=0.01),
        gd.components.Slider(label="Concavity Worst", minimum=0, maximum=1, step=0.01),
        gd.components.Slider(label="Concave Points Worst", minimum=0, maximum=1, step=0.01),
    ],
    outputs="text",
    title="Breast Cancer Prediction",
    description="Enter the features of breast cancer to predict whether the condition is benign or malignant.",
    theme="compact" 
)

# Launch the interface
iface.launch(share=True)
