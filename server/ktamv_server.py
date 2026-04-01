import cv2
import datetime
import requests
import numpy as np
from PIL import Image
from flask import Flask, Response, request

app = Flask(__name__)

# 🔥 FIXE AUFLÖSUNG
_FRAME_WIDTH = 1920
_FRAME_HEIGHT = 1080

# Optionen
_SHOW_TEXT_OVERLAY = False

camera_url = None


def log(msg):
    print(f"[KTAMV] {msg}")


def get_frame():
    global camera_url

    if camera_url is None:
        return None

    try:
        response = requests.get(camera_url, stream=True, timeout=3)
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return None

        # 🔥 AUFLÖSUNG ERZWINGEN
        frame = cv2.resize(frame, (_FRAME_WIDTH, _FRAME_HEIGHT))

        return frame

    except Exception as e:
        log(f"Camera error: {e}")
        return None


def drawOnFrame(frame):
    if not _SHOW_TEXT_OVERLAY:
        return frame

    # optional (aktuell deaktiviert)
    return frame


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = get_frame()

            if frame is None:
                continue

            frame = drawOnFrame(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_server_cfg', methods=['POST'])
def set_server_cfg():
    global camera_url

    data = request.get_json()

    if "camera_url" in data:
        camera_url = data["camera_url"]

    return {"status": "ok"}


if __name__ == '__main__':
    log("Starting KTAMV server (1920x1080)")
    app.run(host='0.0.0.0', port=8080)
