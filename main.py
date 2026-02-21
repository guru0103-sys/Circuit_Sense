import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from inference_sdk import InferenceHTTPClient

# --- DEFAULT CONFIGURATION ---
DEFAULT_MODEL_ID = "electronics-components-cdyjj-wucwm/1"
DEFAULT_MIN_CONFIDENCE = 0.20
RULES_PATH = Path("rules.json")


@st.cache_data
def load_rules(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f).get("circuit_rules", [])


def get_api_key():
    if "ROBOFLOW_API_KEY" in st.secrets:
        return st.secrets["ROBOFLOW_API_KEY"]
    return os.getenv("ROBOFLOW_API_KEY", "")


def build_client(api_key: str):
    return InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=api_key)


def draw_predictions(image, predictions):
    frame = image.copy()
    frame_h, frame_w = frame.shape[:2]

    for pred in predictions:
        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]
        label = pred["class"]
        conf = pred["confidence"]

        x1 = max(0, min(frame_w - 1, int(x - w / 2)))
        y1 = max(0, min(frame_h - 1, int(y - h / 2)))
        x2 = max(0, min(frame_w - 1, int(x + w / 2)))
        y2 = max(0, min(frame_h - 1, int(y + h / 2)))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} ({conf:.2f})",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return frame


def evaluate_rules(predictions, rules):
    detected_parts = [pred["class"].lower() for pred in predictions]
    alerts = []
    for rule in rules:
        setup_present = all(item.lower() in detected_parts for item in rule["setup"])
        if setup_present:
            missing = [req for req in rule["required"] if req.lower() not in detected_parts]
            if missing:
                alerts.append(rule["warning"])
    return alerts


def log_results(alerts, detected_count):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("performance_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Objects Found: {detected_count} | Alerts: {len(alerts)}\n")

    if alerts:
        with open("session_log.txt", "a", encoding="utf-8") as f:
            for alert in alerts:
                f.write(f"[{timestamp}] {alert}\n")


def read_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def main():
    st.set_page_config(page_title="CircuitSense AI Assistant", layout="wide")
    st.title("CircuitSense AI Assistant")
    st.caption("Streamlit web version for hosted deployment")

    api_key = get_api_key()
    if not api_key:
        st.error("Missing API key. Set `ROBOFLOW_API_KEY` in Streamlit secrets or environment variables.")
        st.stop()

    model_id = st.text_input("Model ID", DEFAULT_MODEL_ID)
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, DEFAULT_MIN_CONFIDENCE, 0.01)
    rules = load_rules(RULES_PATH)

    source = st.radio("Image source", ["Upload image", "Use camera"], horizontal=True)
    uploaded = None
    if source == "Upload image":
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    else:
        uploaded = st.camera_input("Take a picture")

    if uploaded is None:
        st.info("Provide an image to run detection.")
        return

    image = read_image(uploaded)
    if image is None:
        st.error("Could not read the image. Try another file.")
        return

    client = build_client(api_key)

    with st.spinner("Running inference..."):
        result = client.infer(image, model_id=model_id)

    raw_predictions = result["predictions"] if isinstance(result, dict) else []
    predictions = [p for p in raw_predictions if float(p.get("confidence", 0.0)) >= min_conf]
    alerts = evaluate_rules(predictions, rules)
    status = "FAULTY" if alerts else "HEALTHY"
    log_results(alerts, len(predictions))

    annotated = draw_predictions(image, predictions)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Status: {status}", use_container_width=True)
    with col2:
        st.metric("Detected Components", len(predictions))
        st.metric("Alerts", len(alerts))
        if status == "HEALTHY":
            st.success("Circuit appears healthy.")
        else:
            st.error("Circuit faults detected.")
            for alert in alerts:
                st.write(f"- {alert}")

    with st.expander("Raw Predictions"):
        if raw_predictions:
            st.json(raw_predictions)
        else:
            st.write("No predictions returned.")


if __name__ == "__main__":
    main()
