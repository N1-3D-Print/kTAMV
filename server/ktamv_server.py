# import the Flask module, the MJPEGResponse class, and the os module
import datetime
import io
import json
import logging
import os
import random
import threading
import time
import traceback
from argparse import ArgumentParser
from dataclasses import dataclass

import matplotlib.font_manager as fm
import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageDraw, ImageFont
from waitress import serve

from ktamv_server_dm import Ktamv_Server_Detection_Manager as dm

__logdebug = ""

# URL to the cloud server
__CLOUD_URL = "http://ktamv.ignat.se/index.php"

# If no nozzle found in this time, timeout the function
__CV_TIMEOUT = 20

# Minimum amount of matches to confirm toolhead position after a move
__CV_MIN_MATCHES = 3

# Default frame size
_FRAME_WIDTH = 800
_FRAME_HEIGHT = 600

# Overlay / preview config
__PREVIEW_FPS = 2
__show_text_overlay = False
__show_detection_overlay = False

# Detection preprocessing config
__gamma_value = 1.2
__preprocess_mode = "default"  # default | soft | off

# If the nozzle position is within this many pixels when comparing frames,
# it's considered a match. Only whole numbers are supported.
__detection_tolerance = 0

# Whether to update the image at next request
__update_static_image = True

# Error message to show on the image
__error_message_to_image = ""

# Indicates if preview is running
__preview_running = False

# Create logs folder if it doesn't exist and configure logging
if not os.path.exists("./logs"):
    os.makedirs("logs")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%a, %d %b %Y %H:%M:%S",
    filename="logs/ktamv_server.log",
    filemode="w",
    encoding="utf-8",
)

# create a Flask app
app = Flask(__name__)

# Define a global variable to store the processed frame in form of an image
__processed_frame_as_image = None

# Define a global variable to store the processed frame in form of bytes
__processed_frame_as_bytes = None

# The loaded standby image
__standby_image = None

# Define a global variable to store the camera path.
_camera_url = None

# Whether to send the frame to the cloud
__send_frame_to_cloud = False

# Define a global variable to store a key-value pair of the request id and the result
request_results = dict()

# The transform matrix calculated from the calibration points
_transformMatrix = None


@dataclass
class Ktamv_Request_Result:
    request_id: int
    data: str  # As JSON encoded string
    runtime: float = None
    statuscode: int = None
    statusmessage: str = None


def clamp_int(value, minimum, maximum, fallback):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, value))


def clamp_float(value, minimum, maximum, fallback):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, value))


def normalize_preprocess_mode(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"default", "soft", "off"}:
        return value
    return None


def build_detection_manager(send_to_cloud):
    return dm(
        log=log,
        camera_url=_camera_url,
        cloud_url=__CLOUD_URL if send_to_cloud else "",
        send_to_cloud=send_to_cloud,
        show_detection_overlay=__show_detection_overlay,
        gamma_value=__gamma_value,
        preprocess_mode=__preprocess_mode,
    )


# Returns the transposed matrix calculated from the calibration points
@app.route("/calculate_camera_to_space_matrix", methods=["POST"])
def calculate_camera_to_space_matrix():
    show_error_message_to_image("")
    try:
        log("*** calling calculate_camera_to_space_matrix ***")

        try:
            data = json.loads(request.data)
            calibration_points = data.get("calibration_points")
        except json.JSONDecodeError:
            return "JSON Decode Error", 400

        if calibration_points is None:
            return "Calibration Points not found in JSON", 400

        n = len(calibration_points)
        real_coords, pixel_coords = np.empty((n, 2)), np.empty((n, 2))
        for i, (r, p) in enumerate(calibration_points):
            real_coords[i] = r
            pixel_coords[i] = p

        x, y = pixel_coords[:, 0], pixel_coords[:, 1]
        a_matrix = np.vstack([x**2, y**2, x * y, x, y, np.ones(n)]).T
        transform = np.linalg.lstsq(a_matrix, real_coords, rcond=None)

        global _transformMatrix
        _transformMatrix = transform[0].T
        return "OK", 200
    except Exception as e:
        show_error_message_to_image("Error: Could not calculate image to space matrix.")
        log("Error: " + str(e) + "<br>" + str(traceback.format_exc()))
        return ""


@app.route("/calculate_offset_from_matrix", methods=["POST"])
def calculate_offset_from_matrix():
    show_error_message_to_image("")
    try:
        log("*** calling calculate_offset ***")
        try:
            data = json.loads(request.data)
            _v = data.get("_v")
            log("_v: " + str(_v))
            log("_transformMatrix: " + str(_transformMatrix))
        except json.JSONDecodeError:
            log("JSON Decode Error")
            return "JSON Decode Error", 400

        offsets = -1 * (0.55 * _transformMatrix @ _v)
        return jsonify(offsets.tolist())
    except Exception as e:
        show_error_message_to_image("Error: Could not calculate offset from matrix.")
        log("Error: " + str(e) + "<br>" + str(traceback.format_exc()))


@app.route("/set_server_cfg", methods=["POST"])
def set_server_cfg():
    show_error_message_to_image("")
    try:
        log("*** calling set_server_cfg ***")
        response_lines = []

        global __preview_running
        global __detection_tolerance
        global __send_frame_to_cloud
        global _camera_url
        global _FRAME_WIDTH
        global _FRAME_HEIGHT
        global __show_text_overlay
        global __show_detection_overlay
        global __gamma_value
        global __preprocess_mode

        __preview_running = False

        try:
            data = json.loads(request.data)
        except json.JSONDecodeError:
            show_error_message_to_image("Error: Could not set server config.")
            return "JSON Decode Error", 400

        camera_url = data.get("camera_url")
        if camera_url is None:
            show_error_message_to_image("Error: Could not set camera URL.")
            return "Camera path not found in JSON", 400

        if not (
            camera_url.casefold().startswith("http://")
            or camera_url.casefold().startswith("https://")
        ):
            show_error_message_to_image("Error: Invalid nozzle_cam_url.")
            log("*** end of set_server_cfg (not set) ***<br>")
            return "Camera path must start with http:// or https://", 400

        _camera_url = camera_url
        response_lines.append("Camera path set to " + _camera_url)

        send_frame_to_cloud = data.get("send_frame_to_cloud")
        if send_frame_to_cloud is not None:
            __send_frame_to_cloud = bool(send_frame_to_cloud)
            response_lines.append(
                f"send_frame_to_cloud set to {__send_frame_to_cloud}"
            )

        detection_tolerance = data.get("detection_tolerance")
        if detection_tolerance is not None:
            __detection_tolerance = clamp_int(
                detection_tolerance, 0, 1000, __detection_tolerance
            )
            response_lines.append(
                f"detection_tolerance set to {__detection_tolerance}"
            )

        frame_width = data.get("frame_width")
        if frame_width is not None:
            _FRAME_WIDTH = clamp_int(frame_width, 64, 4096, _FRAME_WIDTH)
            response_lines.append(f"frame_width set to {_FRAME_WIDTH}")

        frame_height = data.get("frame_height")
        if frame_height is not None:
            _FRAME_HEIGHT = clamp_int(frame_height, 64, 4096, _FRAME_HEIGHT)
            response_lines.append(f"frame_height set to {_FRAME_HEIGHT}")

        show_text_overlay = data.get("show_text_overlay")
        if show_text_overlay is not None:
            __show_text_overlay = bool(show_text_overlay)
            response_lines.append(
                f"show_text_overlay set to {__show_text_overlay}"
            )

        show_detection_overlay = data.get("show_detection_overlay")
        if show_detection_overlay is not None:
            __show_detection_overlay = bool(show_detection_overlay)
            response_lines.append(
                f"show_detection_overlay set to {__show_detection_overlay}"
            )

        gamma_value = data.get("gamma_value")
        if gamma_value is not None:
            __gamma_value = clamp_float(gamma_value, 0.1, 5.0, __gamma_value)
            response_lines.append(f"gamma_value set to {__gamma_value}")

        preprocess_mode = normalize_preprocess_mode(data.get("preprocess_mode"))
        if preprocess_mode is not None:
            __preprocess_mode = preprocess_mode
            response_lines.append(f"preprocess_mode set to {__preprocess_mode}")

        show_error_message_to_image("Camera url set.")
        log(f"*** end of set_server_cfg (set to {_camera_url}) ***<br>")
        return "\n".join(response_lines), 200

    except Exception as e:
        show_error_message_to_image("Error: Could not set camera URL.")
        log("Error: " + str(e) + "<br>" + str(traceback.format_exc()))


# Called from DetectionManager to put the frame in the global variable so it can be sent to the web browser
def put_frame(frame):
    try:
        global __processed_frame_as_image, __update_static_image
        __processed_frame_as_image = Image.fromarray(frame)
        __update_static_image = True
    except Exception as e:
        log("Error: " + str(e) + "<br>" + str(traceback.format_exc()))


@app.route("/getAllReqests")
def getAllReqests():
    try:
        return jsonify(request_results)
    except Exception as e:
        log("Error: " + str(e) + "<br>" + str(traceback.format_exc()))


@app.route("/")
def index():
    file_path = "logs/ktamv_server.log"
    content = "<H1>kTAMV Server is running</H1><br><b>Log file:</b><br>"
    content += (
        f"Frame width: {_FRAME_WIDTH}, Frame height: {_FRAME_HEIGHT}<br>"
        f"Text overlay: {__show_text_overlay}<br>"
        f"Detection overlay: {__show_detection_overlay}<br>"
        f"Gamma: {__gamma_value}<br>"
        f"Preprocess mode: {__preprocess_mode}<br>"
    )
    content += "Debuging log:<br>" + __logdebug + "<br>"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content += file.read()
            content = content.replace("\n", "<br>")
            html_content = (
                f'<html><head><meta charset="utf-8"></head><body>{content}</body></html>'
            )
            return html_content
    except FileNotFoundError:
        return content + "Log file not found"


@app.route("/getReqest", methods=["GET", "POST"])
def getReqest():
    try:
        request_id = request.args.get("request_id", type=int, default=None)
        try:
            return jsonify(request_results[request_id])
        except KeyError:
            return jsonify(
                Ktamv_Request_Result(
                    request_id, None, None, 404, "Request not found"
                )
            )
    except Exception as e:
        log("Error: " + str(e) + "<br>" + str(traceback.format_exc()))


@app.route("/getNozzlePosition")
def getNozzlePosition():
    show_error_message_to_image("")

    global __preview_running
    __preview_running = False

    try:
        log("*** calling getNozzlePosition ***")
        start_time = time.time()
        request_id = random.randint(0, 1000000)

        if _camera_url is None:
            request_results[request_id] = Ktamv_Request_Result(
                request_id, None, time.time() - start_time, 502, "Camera URL not set"
            )
            log("*** end of getNozzlePosition - Camera URL not set ***<br>")
            return jsonify(request_results[request_id])

        request_results[request_id] = Ktamv_Request_Result(
            request_id, None, None, 202, "Accepted"
        )
        log("request_results: " + str(request_results))

        def do_work():
            log("*** calling do_work ***")
            detection_manager = build_detection_manager(__send_frame_to_cloud)

            position = detection_manager.recursively_find_nozzle_position(
                put_frame, __CV_MIN_MATCHES, __CV_TIMEOUT, __detection_tolerance
            )

            log("position: " + str(position))

            if position is None:
                request_result_object = Ktamv_Request_Result(
                    request_id, None, time.time() - start_time, 404, "No nozzle found"
                )
                show_error_message_to_image("Error: No nozzle found.")
            else:
                request_result_object = Ktamv_Request_Result(
                    request_id,
                    json.dumps(position),
                    time.time() - start_time,
                    200,
                    "OK",
                )

            global request_results
            request_results[request_id] = request_result_object
            log("*** end of do_work ***")

        thread = threading.Thread(target=do_work)
        thread.start()

        log("*** end of getNozzlePosition ***<br>")
        return jsonify(request_results[request_id])
    except Exception as e:
        show_error_message_to_image("Error: Could not get nozzle position.")
        log("Error: " + str(e) + "<br>" + str(traceback.format_exc()))


@app.route("/preview", methods=["POST"])
def preview():
    show_error_message_to_image("")
    try:
        log("*** calling preview ***")
        global __preview_running

        try:
            data = json.loads(request.data)
            action = data.get("action")
        except json.JSONDecodeError:
            show_error_message_to_image("Error: Could not get action.")
            return "JSON Decode Error", 400

        def do_preview():
            log("*** calling do_preview ***")
            detection_manager = build_detection_manager(False)

            while __preview_running:
                detection_manager.get_preview_frame(put_frame)
                time.sleep(1 / __PREVIEW_FPS)

            log("*** end of do_preview ***")

        if action == "stop":
            __preview_running = False
            return "Stopped preview.", 200
        if action == "start":
            if _camera_url is None:
                log("*** end of preview - Camera URL not set ***<br>")
                return "Camera URL not set", 502

            __preview_running = True
            thread = threading.Thread(target=do_preview)
            thread.start()
            return "Started preview.", 200

        return "Invalid action.", 400
    except Exception as e:
        show_error_message_to_image("Error: Could not do preview.")
        log("Error: " + str(e) + "<br>" + str(traceback.format_exc()))


###
# Returns the image to the web browser to act as a webcam
###
@app.route("/image")
def image():
    try:
        global __processed_frame_as_bytes
        global __update_static_image
        global __standby_image
        global __processed_frame_as_image

        if __processed_frame_as_image is None:
            __processed_frame_as_image = Image.open("standby.jpg", mode="r")
            __processed_frame_as_image.load()
            __update_static_image = True

        if __update_static_image:
            __update_static_image = False
            rendered_frame = drawOnFrame(__processed_frame_as_image.copy())
            img_io = io.BytesIO()
            rendered_frame.save(img_io, "JPEG")
            img_io.seek(0)
            __processed_frame_as_bytes = img_io.read()

        processed_frame_file = io.BytesIO(__processed_frame_as_bytes)
        processed_frame_file.seek(0)
        return send_file(processed_frame_file, mimetype="image/jpeg")
    except Exception as e:
        log("Error: " + str(e) + "<br>" + str(traceback.format_exc()))


def drawOnFrame(usedFrame):
    if not __show_text_overlay:
        return usedFrame

    current_datetime = datetime.datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")
    usedFrame = drawTextOnFrame(usedFrame, "Updated: " + current_datetime_str, row=1)

    if _camera_url is None:
        usedFrame = drawTextOnFrame(
            usedFrame, "kTAMV Server Configuration not recieved.", row=2
        )
    elif __processed_frame_as_image is None:
        usedFrame = drawTextOnFrame(usedFrame, "No image recieved since start.", row=2)
    elif _transformMatrix is None:
        usedFrame = drawTextOnFrame(usedFrame, "Camera not calibrated.", row=2)

    if __error_message_to_image != "":
        usedFrame = drawTextOnFrame(usedFrame, __error_message_to_image, row=3)

    if __preview_running:
        usedFrame = drawTextOnFrame(
            usedFrame, "Preview running.", row=-1, row_width=270
        )

    return usedFrame


def drawTextOnFrame(usedFrame, text, row=1, row_width=None):
    try:
        font_size = 28
        font_color = (255, 255, 255)
        first_row_start = (10, 10)

        draw = ImageDraw.Draw(usedFrame)
        font_path = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
        font = ImageFont.truetype(font_path, font_size)

        if row > 0:
            start_point = (
                first_row_start[0],
                first_row_start[1] + (row - 1) * (font_size + 10),
            )
        else:
            start_point = (
                first_row_start[0],
                usedFrame.height - (abs(row) * (font_size + 10) + first_row_start[1]),
            )

        if row_width is None:
            row_width = usedFrame.width - 10

        draw.rectangle(
            (
                start_point[0] - 5,
                start_point[1] - 5,
                row_width,
                start_point[1] + font_size + 10,
            ),
            fill=(0, 0, 0),
        )
        draw.text(start_point, text, font=font, fill=font_color)
        return usedFrame
    except Exception as e:
        log("Error: " + str(e) + "<br>" + str(traceback.format_exc()))
        return usedFrame


def log_clear():
    global __logdebug
    __logdebug = ""


def log(message: str):
    global __logdebug
    __logdebug += message + "<br>"


def log_get():
    global __logdebug
    return __logdebug


def show_error_message_to_image(message: str):
    global __error_message_to_image, __update_static_image
    __error_message_to_image = message
    __update_static_image = True


# Run the app on the specified port
if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=8085, help="Port number")
    args = parser.parse_args()

    serve(app, host="0.0.0.0", port=args.port)
