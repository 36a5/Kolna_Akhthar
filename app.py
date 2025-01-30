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
import base64




@st.cache_resource
def get_data_and_get_model():
    return YOLO(r".\utils\models\potholes_model\best.pt")
    

def set_bg_and_css():
 
    with open(r".\utils\style\background.png", "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    
    # تحميل CSS الخارجي
    with open(r".\utils\style\style.css", "r", encoding="utf-8") as f:
        css = f.read()
    
    # دمج الاثنين
    custom_style = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;700&display=swap');

        {css}
         .folium-map {{
            border-radius: 20px !important;
            box-shadow: 0 20px 30px rgba(0,0,0,0.2) !important;
            border: 3px solid #ffffff !important;
        }}
        
         .leaflet-popup-content-wrapper {{
            border-radius: 200px !important;
            background: #f8f9fa !important;
        }}
        
        div[data-testid="stVerticalBlock"] > button[kind="header"] {{
            background: #e74c3c !important;
            color: white !important;
        }}
        
        
        .stApp {{
            background: url("data:image/png;base64,{b64}") !important;

        }}
        </style>
    """
    st.markdown(custom_style, unsafe_allow_html=True)

set_bg_and_css()


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



def on_click():
    st.session_state.run_camera = True

if "x" not in st.session_state:
    if os.listdir("utils/public/images"):
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

m = folium.Map(location=[24.564751, 46.904795], zoom_start=10)

for _, row in data.iterrows():
    popup_html = f"""
    {row['lat']},{row['lon']}
    {row['image']}
    """
    folium.Marker(
    location=[row['lat'], row['lon']],
    popup=folium.Popup(popup_html, max_width=400),
    icon=folium.Icon(
        icon='map-marker-alt',
        prefix='fa',
        icon_color='green',
        icon_size=(10, 10),
    )
).add_to(m)
    
m_col1, m_col2 = st.columns(2)
with m_col1:    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Capture", on_click=on_click):
            pass

    with col2:
        if st.session_state.get("run_camera", False):
            if st.button("Close Camera"):
                st.session_state.run_camera = False

    camera()
with m_col2:
    if st.session_state["show_map"] == False:
        st.button("View Map", on_click=toggle_map)
    else:
        st.button("Close Map", on_click=toggle_map)
        
        st_folium(m, width=680,
                    height=680
                    ,returned_objects=[],
                     key='enhanced_map')
                    

    with open(r"utils\data\pins.pkl", "wb") as file:
        pickle.dump(pins, file)
        
        
        
