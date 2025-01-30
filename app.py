from ultralytics import YOLO
import streamlit as st
import pandas as pd
import cv2
import time
from PIL import Image
import base64
from io import BytesIO
import folium
from streamlit_folium import st_folium
import pickle
from streamlit_js_eval import get_geolocation
import os


@st.cache_resource
def get_data_and_get_model():
    return YOLO(r"utils\models\potholes_model\best.pt")

def load_pins():
    file_path = r"utils\data\pins.pkl"
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        pins = []
    else:
        with open(file_path, "rb") as file:
            pins = pickle.load(file)
    return pins


def pridect_ai(model, img: str):
    try:
        results = model(img)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{results[0].names[int(box.cls)]} {float(box.conf):.1}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            detections.append({
                "class": results[0].names[int(box.cls)],
                "confidence": float(box.conf),
                "coordinates": box.xywh.tolist()[0]
            })
        if not detections:
            return detections, img, False
        else:
            location = get_geolocation()
            if location:
                lat, lon = location["coords"]["latitude"], location["coords"]["longitude"]
                pins.append({"lat": lat, "lon": lon, "image_path": f"utils/public/images/{st.session_state.x}.jpg"})
                with open(r"utils\data\pins.pkl", "wb") as file:
                    pickle.dump(pins, file)
            else:
                st.write("Location access not granted or unavailable.")
            return detections, img , True
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None , False

pins = load_pins()
model = get_data_and_get_model()

st.title("Hello my website")

def on_click():
    st.session_state.run_camera = True

if "x" not in st.session_state:
    if os.listdir("utils/public/images")[-1].split(".")[0].isdigit():
        st.session_state.x = int(os.listdir("utils/public/images")[-1].split(".")[0])
    else:
        st.session_state.x = 0

if "capture" not in st.session_state:
    st.session_state["capture"] = []

capture = st.session_state["capture"]

def camera():
    if "run_camera" in st.session_state and st.session_state.run_camera:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        
        if ret:
            detections, processed_image, isdetected = pridect_ai(model, frame)
            if processed_image is not None and isdetected:
                st.session_state.x += 1
                cv2.imwrite(f"utils/public/images/{st.session_state.x}.jpg", processed_image)

            st.image(frame, channels="BGR")
            st.session_state.capture.append(processed_image)
        time.sleep(0.5)
        st.rerun()

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Capture", on_click=on_click):
        pass

with col2:
    if st.session_state.get("run_camera", False):
        if st.button("Close Camera"):
            st.session_state.run_camera = False

camera()

if "show_map" not in st.session_state:
    st.session_state["show_map"] = False

def toggle_map():
    st.session_state["show_map"] = not st.session_state["show_map"]

def image_popup(image_path):
    try:
        img = Image.open(image_path)
        img.thumbnail((200, 200))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        html = f'<img src="data:image/jpeg;base64,{img_str}" alt="Point Image">'
        return html
    except Exception as e:
        return f"Error displaying image: {e}"

data = pd.DataFrame(pins)
if data.empty!=True:
        data['image'] = data['image_path'].apply(image_popup)

m = folium.Map(location=[24.7136, 46.6753], zoom_start=5)

for _, row in data.iterrows():
    popup_html = f"""
    {row['lat']},{row['lon']}
    {row['image']}
    """
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(m)

if st.session_state["show_map"] == False:
    st.button("View Map", on_click=toggle_map)
else:
    st.button("Close Map", on_click=toggle_map)
    st_folium(m, width=700, height=500)

with open(r"utils\data\pins.pkl", "wb") as file:
    pickle.dump(pins, file)
