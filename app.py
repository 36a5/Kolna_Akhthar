from ultralytics import YOLO
import streamlit as st
import pandas as pd
import cv2
import time

@st.cache_resource
def my_model():
    return YOLO(r"utils\models\potholes_model\best.pt")


latitude1 =24.790704
longitude1 =46.585779


def pridect_ai(model,img:str):
    try:
    
        results=model(img)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            label=f"{results[0].names[int(box.cls)]} {float(box.conf):.1}"
            cv2.putText(img,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            
            detections.append({
                "class": results[0].names[int(box.cls)],
                "confidence": float(box.conf),
                "coordinates": box.xywh.tolist()[0]
            })
        return detections,img
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None,None
                 
            
model=my_model()

st.title("Hello my website")

def on_click():
    st.session_state.run_camera = True
    
if "x" not in st.session_state:
    st.session_state.x = 0
    
if "capture" not in st.session_state:
    st.session_state["capture"]=[]
    
capture=st.session_state["capture"]

def camera():
    if "run_camera" in st.session_state and st.session_state.run_camera:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        
        if ret:
            st.session_state.x +=1
            detections, processed_image = pridect_ai(model, frame)
            if processed_image is not None:
                cv2.imwrite(f"utils/public/images{st.session_state.x}.jpg",processed_image)

            st.image(frame, channels="BGR")
            st.session_state.capture.append(processed_image)
        time.sleep(0.5)
        st.rerun() 
    
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Capture", on_click=on_click):
        pass

with col2:
    if st.session_state.get("run_camera", False) :
        if st.button("Close Camera"):
            st.session_state.run_camera = False

camera()

    
def new_point(latitude:float,longitude:float):
        return pd.DataFrame({
            'lat': [latitude],
            'lon': [longitude],
            'size':[1.0],
            
            }
        )

show_map=st.session_state.get("show_map",False)


def View_map(boolean:bool)->bool:
    st.session_state.show_map=boolean
    
    
if show_map==False:      
    st.button("View Map",on_click=View_map(True))

elif show_map:  
    st.button("Close Map",on_click=View_map(False))
    st.map(new_point(24.790704,46.585779))
    
