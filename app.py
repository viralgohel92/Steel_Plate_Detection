import streamlit as st
import numpy as np
import cv2
import pandas as pd
import mysql.connector
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime

# === Load the trained model and classes ===
model = load_model("steel_defect_model.h5")
class_names = np.load("classes.npy")

# === Fetch prediction history from MySQL ===
def fetch_prediction_data():
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",  # Use your actual password here
            database="steel_db"
        )
        query = "SELECT image_name, predicted_class, confidence, timestamp FROM defect_log ORDER BY timestamp DESC"
        df = pd.read_sql(query, con=db)
        db.close()
        return df
    except mysql.connector.Error as e:
        st.error(f"Failed to load prediction history: {e}")
        return pd.DataFrame()

# === Streamlit UI Layout ===
st.set_page_config(page_title="Steel Defect Detector", layout="centered")
st.title("🧠 Steel Plate Defect Detection System")

# === Initialize webcam toggle ===
if "camera_used" not in st.session_state:
    st.session_state.camera_used = False

# === Create Tabs ===
tab1, tab2 = st.tabs(["🔍 Detection", "📊 Admin Dashboard"])

# === Tab 1: Detection ===
with tab1:
    st.header("🔍 Upload or Capture Steel Plate Defect")

    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

    st.subheader("📷 Or capture from webcam")

    if not st.session_state.camera_used:
        camera_image = st.camera_input("Take a photo using your webcam")
        if camera_image is not None:
            uploaded_file = camera_image
            st.session_state.camera_used = True
    else:
        st.info("✅ Camera input completed.")
        if st.button("🔄 Retake Photo"):
            st.session_state.camera_used = False

    if uploaded_file is not None:
        file_name = getattr(uploaded_file, 'name', 'captured_image.jpg')

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        image_resized = cv2.resize(image, (128, 128))
        image_resized = image_resized / 255.0
        image_reshaped = image_resized.reshape(1, 128, 128, 1)

        prediction = model.predict(image_reshaped)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        st.success(f"🧾 **Predicted Defect:** {predicted_class}")
        st.info(f"📈 Confidence: {confidence:.2f}%")

        # === Save to MySQL ===
        try:
            db = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",  # Your MySQL password
                database="steel_db"
            )
            cursor = db.cursor()
            query = "INSERT INTO defect_log (image_name, predicted_class, confidence) VALUES (%s, %s, %s)"
            cursor.execute(query, (str(file_name), str(predicted_class), float(confidence)))
            db.commit()
            cursor.close()
            db.close()
            st.success("📦 Prediction saved to database!")
        except Exception as e:
            st.error(f"❌ Failed to save to database: {e}")

# === Tab 2: Admin Dashboard ===
with tab2:
    st.header("📊 Admin Dashboard - Prediction History")

    df = fetch_prediction_data()

    if not df.empty:
        st.subheader("📋 Recent Predictions")
        st.dataframe(df)

        st.subheader("📈 Confidence Trend Over Time")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values('timestamp')
        st.line_chart(df_sorted.set_index('timestamp')['confidence'])

        # CSV Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction Report (CSV)",
            data=csv,
            file_name="steel_defect_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No prediction history found yet.")
