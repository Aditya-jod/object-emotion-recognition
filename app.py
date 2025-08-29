from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from deepface import DeepFace
import threading
import os
import logging

# ---------------- Logging Setup ---------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ObjectEmotionApp")

# ---------------- Flask App ---------------- #
app = Flask(__name__)

# ---------------- LED Setup ---------------- #
try:
    from gpiozero import RGBLED
    led_available = True
    led = RGBLED(red=17, green=27, blue=22)
except ImportError:
    led_available = False

    class DummyLED:
        def off(self): pass
        @property
        def color(self): return (0, 0, 0)
        @color.setter
        def color(self, value): pass

    led = DummyLED()

# ---------------- Face Detector ---------------- #
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- YOLO Setup ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
YOLO_CFG = os.path.join(DATA_DIR, "yolov3.cfg")
YOLO_WEIGHTS = os.path.join(DATA_DIR, "yolov3.weights")
YOLO_NAMES = os.path.join(DATA_DIR, "coco.names")

if not (os.path.exists(YOLO_CFG) and os.path.exists(YOLO_WEIGHTS) and os.path.exists(YOLO_NAMES)):
    logger.error("YOLO files not found! Place yolov3.weights, yolov3.cfg, coco.names in data/ folder.")
    raise FileNotFoundError("Missing YOLO files")


net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
with open(YOLO_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ---------------- Camera Management ---------------- #
camera = None
is_camera_running = False
frame_lock = threading.Lock()


def set_led_color(emotion):
    """Map detected emotion to LED color"""
    if not led_available:
        return
    color_map = {
        "happy": (0, 1, 0),
        "sad": (0, 0, 1),
        "angry": (1, 0, 0),
        "surprise": (1, 1, 0),
        "neutral": (1, 1, 1),
    }
    led.color = color_map.get(emotion, (0, 0, 0))


def start_camera():
    global camera, is_camera_running
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logger.error("Could not open camera")
            return
    is_camera_running = True
    logger.info("Camera started")


def stop_camera():
    global camera, is_camera_running
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None
    is_camera_running = False
    led.off()
    logger.info("Camera stopped")


# ---------------- Frame Processing ---------------- #
def generate_frames():
    global camera, is_camera_running
    while True:
        if is_camera_running and camera is not None and camera.isOpened():
            success, frame = camera.read()
            if not success:
                logger.warning("Frame capture failed, restarting camera")
                stop_camera()
                start_camera()
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                try:
                    analysis = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    dominant_emotion = analysis.get("dominant_emotion", "neutral")
                except Exception as e:
                    logger.error(f"DeepFace error: {e}")
                    dominant_emotion = "neutral"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                set_led_color(dominant_emotion)

            # ---------------- YOLO Object Detection ---------------- #
            try:
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416),
                                             (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)
            except Exception as e:
                logger.error(f"YOLO forward error: {e}")
                continue

            class_ids, confidences, boxes = [], [], []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x, center_y = int(detection[0] * frame.shape[1]), int(detection[1] * frame.shape[0])
                        w, h = int(detection[2] * frame.shape[1]), int(detection[3] * frame.shape[0])
                        x, y = int(center_x - w / 2), int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            
            if len(indexes) > 0:
                if isinstance(indexes, (tuple, list)):
                    indexes = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indexes]
                else:
                    indexes = indexes.flatten().tolist()

                for i in indexes:
                    x, y, w, h = boxes[i]
                    label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Save detection details to a text file
                    detections_file = os.path.join(DATA_DIR, "detections.txt")
                    with open(detections_file, "a") as f:
                       f.write(f"{label} at x={x}, y={y}, w={w}, h={h}\n")

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        else:
            # If camera is stopped, just wait instead of sending blanks
            import time
            time.sleep(0.1)

# ---------------- Flask Routes ---------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start", methods=["POST"])
def start():
    start_camera()
    return "Camera started"


@app.route("/stop", methods=["POST"])
def stop():
    stop_camera()
    return "Camera stopped"


# ---------------- Main ---------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)  # Debug=False for stability
