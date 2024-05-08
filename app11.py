import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from skimage import exposure
from PIL import Image
import io
import matplotlib.pyplot as plt
# Load the pre-trained model
model = tf.keras.models.load_model('/Users/Suchitjain/Desktop/Streamlit/model.h5')
# Define the labels for classification
LABELS = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA', 'COVID19']
# Dictionary containing symptoms and precautions for each disease
disease_info = {
    'NORMAL': {
        'Symptoms': 'No specific symptoms. Normal chest X-ray.',
        'Precautions': 'Maintain a healthy lifestyle. Follow regular check-ups.'
    },
    'TUBERCULOSIS': {
        'Symptoms': 'Persistent cough, weight loss, fatigue, fever, night sweats, coughing up blood.',
        'Precautions': 'Take prescribed antibiotics. Cover mouth while coughing/sneezing. Good hygiene practices.'
    },
    'PNEUMONIA': {
        'Symptoms': 'Cough, fever, difficulty breathing, chest pain, fatigue, nausea/vomiting.',
        'Precautions': 'Get vaccinated. Wash hands regularly. Avoid close contact with sick individuals.'
    },
    'COVID19': {
        'Symptoms': 'Fever, cough, shortness of breath, fatigue, body aches, loss of taste or smell.',
        'Precautions': 'Wear masks. Practice social distancing. Wash hands frequently. Get vaccinated.'
    }
}
# Function to predict label given an image
def predict(input_image):
    # Convert the image to 8-bit unsigned integer if it's of type CV_64F
    if input_image.dtype == np.float64:
        input_image = (input_image * 255).astype(np.uint8)
    # Ensure the image is in RGB format
    if len(input_image.shape) == 2: # Grayscale image
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    elif input_image.shape[2] == 1: # Single channel image (e.g., grayscale)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    # Resize the image to (256, 256) as expected by the model
    input_image = cv2.resize(input_image, (256, 256))
    # Normalize the image
    img_4d = input_image.reshape(-1, 256, 256, 3) / 255.0
    # Make the prediction
    prediction = model.predict(img_4d)
    # Check if prediction is empty or has insufficient elements
    if prediction.size == 0:
        return {}
    # Convert predictions to percentage format
    total_sum = sum(prediction[0])
    prediction_percentage = {LABELS[i]: round((prediction[0][i] / total_sum) * 100, 2) for i in range(min(len(prediction[0]), len(LABELS)))}
    
    return prediction_percentage

def noise_filter(input_image):
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.medianBlur(image, 3)
    equalized_image = exposure.equalize_hist(denoised_image)
    blurred_image = cv2.GaussianBlur(equalized_image, (3, 3), 0)
    sharpened_image = cv2.addWeighted(equalized_image, 1.5, blurred_image, -0.5, 0)
    normalized_image = (sharpened_image - sharpened_image.min()) / (sharpened_image.max() - sharpened_image.min())
    resized_image = cv2.resize(normalized_image, (200, 200))
    return resized_image

# Function to download the filtered image
def download_filtered_image(image_array):
    img_pil = Image.fromarray((image_array * 255).astype(np.uint8))  # Convert to 8-bit unsigned integer
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

# Set up the Streamlit interface
st.title('SKAI')
uploaded_image = st.file_uploader("Upload an X-ray image", type=["jpg", "JPEG", "png"])
if uploaded_image is not None:
    # Convert the uploaded file to an image
    image = Image.open(io.BytesIO(uploaded_image.getvalue()))
    # Convert the image to a NumPy array
    image_array = np.array(image)
    # Process the image
    filtered_image = noise_filter(image_array)
    # Display the filtered image
    st.subheader('Filtered Image')
    st.image(filtered_image, use_column_width=True)    
    # Classify the uploaded image
    prediction_result = predict(filtered_image)
    # Display prediction results
    st.subheader('Prediction Results:')
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    for label, percentage in prediction_result.items():
        st.markdown(f'<p><span class="prediction-label">{label}:</span> {percentage}%</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Find the disease with the highest predicted probability
    predicted_label = max(prediction_result, key=prediction_result.get)

    # Display detailed information based on the predicted label
    st.sidebar.subheader(f'Detailed Overview: {predicted_label}')
    st.sidebar.write(f"Symptoms: {disease_info[predicted_label]['Symptoms']}")
    st.sidebar.write(f"Precautions: {disease_info[predicted_label]['Precautions']}")

    # Button to download the filtered image
    if st.button('Download Filtered Image'):
        filtered_img_bytes = download_filtered_image(filtered_image)
        st.download_button(label='Download Filtered Image', data=filtered_img_bytes, file_name='filtered_image.png', mime='image/png', key=None)

    # Generate a pie chart based on the prediction percentages
    st.subheader('Prediction Chart:')
    fig, ax = plt.subplots()
    ax.bar(prediction_result.keys(), prediction_result.values())
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Label')
    st.pyplot(fig)

