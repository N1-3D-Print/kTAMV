import copy
import time

import cv2
import numpy as np

from ktamv_server_io import Ktamv_Server_Io as io


class Ktamv_Server_Detection_Manager:
    uv = [None, None]
    __algorithm = None
    __io = None

    def __init__(
        self,
        log,
        camera_url,
        cloud_url,
        send_to_cloud=False,
        show_detection_overlay=False,
        gamma_value=1.2,
        preprocess_mode="default",
        *args,
        **kwargs,
    ):
        try:
            self.log = log
            self.log("*** calling DetectionManager.__init__")

            self.send_to_cloud = send_to_cloud
            self.show_detection_overlay = bool(show_detection_overlay)
            self.gamma_value = float(gamma_value)
            self.preprocess_mode = str(preprocess_mode).strip().lower()

            self.__io = io(
                log=log,
                camera_url=camera_url,
                cloud_url=cloud_url,
                save_image=False,
            )

            self.__algorithm = None
            self.createDetectors()

            self.log("*** exiting DetectionManager.__init__")
        except Exception as e:
            self.log("*** exception in DetectionManager.__init__: %s" % str(e))
            raise e

    # timeout = 20: If no nozzle found in this time, timeout the function
    # min_matches = 3: Minimum amount of matches to confirm toolhead position after a move
    # xy_tolerance = 1: If the nozzle position is within this tolerance, it's considered a match.
    def recursively_find_nozzle_position(
        self, put_frame_func, min_matches, timeout, xy_tolerance
    ):
        self.log("*** calling recursively_find_nozzle_position")
        start_time = time.time()
        last_pos = (0, 0)
        pos_matches = 0
        pos = None

        while time.time() - start_time < timeout:
            frame = self.__io.get_single_frame()
            positions, processed_frame = self.nozzleDetection(frame)
            if processed_frame is not None:
                put_frame_func(processed_frame)

            self.log("recursively_find_nozzle_position positions: %s" % str(positions))

            if positions is None or len(positions) == 0:
                continue

            pos = positions
            if (
                abs(pos[0] - last_pos[0]) <= xy_tolerance
                and abs(pos[1] - last_pos[1]) <= xy_tolerance
            ):
                pos_matches += 1
                if pos_matches >= min_matches:
                    self.log(
                        "recursively_find_nozzle_position found %i matches and returning"
                        % pos_matches
                    )
                    if self.send_to_cloud:
                        self.__io.send_frame_to_cloud(frame, pos, self.__algorithm)
                    break
            else:
                self.log(
                    "Position found does not match last position. Last position: %s, current position: %s"
                    % (str(last_pos), str(pos))
                )
                self.log(
                    "Difference: X%.3f Y%.3f"
                    % (abs(pos[0] - last_pos[0]), abs(pos[1] - last_pos[1]))
                )
                pos_matches = 0

            last_pos = pos
            time.sleep(0.3)

        self.log("recursively_find_nozzle_position found: %s" % str(last_pos))
        self.log("*** exiting recursively_find_nozzle_position")
        return pos

    def get_preview_frame(self, put_frame_func):
        frame = self.__io.get_single_frame()
        _, processed_frame = self.nozzleDetection(frame)
        if processed_frame is not None:
            put_frame_func(processed_frame)
        return

    # ----------------- TAMV Nozzle Detection as tested in ktamv_cv -----------------

    def createDetectors(self):
        self.standardParams = cv2.SimpleBlobDetector_Params()
        self.standardParams.minThreshold = 1
        self.standardParams.maxThreshold = 50
        self.standardParams.thresholdStep = 1
        self.standardParams.filterByArea = True
        self.standardParams.minArea = 400
        self.standardParams.maxArea = 900
        self.standardParams.filterByCircularity = True
        self.standardParams.minCircularity = 0.8
        self.standardParams.maxCircularity = 1
        self.standardParams.filterByConvexity = True
        self.standardParams.minConvexity = 0.3
        self.standardParams.maxConvexity = 1
        self.standardParams.filterByInertia = True
        self.standardParams.minInertiaRatio = 0.3

        self.relaxedParams = cv2.SimpleBlobDetector_Params()
        self.relaxedParams.minThreshold = 1
        self.relaxedParams.maxThreshold = 50
        self.relaxedParams.thresholdStep = 1
        self.relaxedParams.filterByArea = True
        self.relaxedParams.minArea = 600
        self.relaxedParams.maxArea = 15000
        self.relaxedParams.filterByCircularity = True
        self.relaxedParams.minCircularity = 0.6
        self.relaxedParams.maxCircularity = 1
        self.relaxedParams.filterByConvexity = True
        self.relaxedParams.minConvexity = 0.1
        self.relaxedParams.maxConvexity = 1
        self.relaxedParams.filterByInertia = True
        self.relaxedParams.minInertiaRatio = 0.3

        t1 = 20
        t2 = 200
        overall = 0.5
        area = 200

        self.superRelaxedParams = cv2.SimpleBlobDetector_Params()
        self.superRelaxedParams.minThreshold = t1
        self.superRelaxedParams.maxThreshold = t2
        self.superRelaxedParams.filterByArea = True
        self.superRelaxedParams.minArea = area
        self.superRelaxedParams.filterByCircularity = True
        self.superRelaxedParams.minCircularity = overall
        self.superRelaxedParams.filterByConvexity = True
        self.superRelaxedParams.minConvexity = overall
        self.superRelaxedParams.filterByInertia = True
        self.superRelaxedParams.minInertiaRatio = overall
        self.superRelaxedParams.filterByColor = False
        self.superRelaxedParams.minDistBetweenBlobs = 2

        self.detector = cv2.SimpleBlobDetector_create(self.standardParams)
        self.relaxedDetector = cv2.SimpleBlobDetector_create(self.relaxedParams)
        self.superRelaxedDetector = cv2.SimpleBlobDetector_create(self.superRelaxedParams)

    def get_frame_geometry(self, frame):
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        return width, height, center

    def draw_detection_overlay(
        self,
        frame,
        center=None,
        keypoint_radius=17,
        keypoint_color=(0, 255, 0),
    ):
        if not self.show_detection_overlay:
            return frame

        output = copy.deepcopy(frame)
        width, height, frame_center = self.get_frame_geometry(output)
        cx, cy = frame_center

        output = cv2.line(output, (cx, 0), (cx, height), (0, 0, 0), 2)
        output = cv2.line(output, (0, cy), (width, cy), (0, 0, 0), 2)
        output = cv2.line(output, (cx, 0), (cx, height), (255, 255, 255), 1)
        output = cv2.line(output, (0, cy), (width, cy), (255, 255, 255), 1)

        if center is None:
            output = cv2.circle(
                img=output,
                center=(cx, cy),
                radius=keypoint_radius,
                color=(0, 0, 0),
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            output = cv2.circle(
                img=output,
                center=(cx, cy),
                radius=keypoint_radius + 1,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            return output

        x, y = int(center[0]), int(center[1])
        circle_frame = cv2.circle(
            img=output,
            center=(x, y),
            radius=keypoint_radius,
            color=keypoint_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        output = cv2.addWeighted(circle_frame, 0.4, output, 0.6, 0)
        output = cv2.circle(
            img=output,
            center=(x, y),
            radius=keypoint_radius,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        output = cv2.line(output, (x - 5, y), (x + 5, y), (255, 255, 255), 2)
        output = cv2.line(output, (x, y - 5), (x, y + 5), (255, 255, 255), 2)
        return output

    def nozzleDetection(self, image):
        nozzleDetectFrame = copy.deepcopy(image)
        displayFrame = copy.deepcopy(image)

        keypoints = None
        center = (None, None)

        if 1 == 1:
            preprocessorImage0 = self.preprocessImage(
                frameInput=nozzleDetectFrame, algorithm=0
            )
            preprocessorImage1 = self.preprocessImage(
                frameInput=nozzleDetectFrame, algorithm=1
            )
            preprocessorImage2 = self.preprocessImage(
                frameInput=nozzleDetectFrame, algorithm=2
            )

            keypoints = self.detector.detect(preprocessorImage0)
            keypointColor = (0, 0, 255)
            if len(keypoints) != 1:
                keypoints = self.detector.detect(preprocessorImage1)
                keypointColor = (0, 255, 0)
                if len(keypoints) != 1:
                    keypoints = self.relaxedDetector.detect(preprocessorImage0)
                    keypointColor = (255, 0, 0)
                    if len(keypoints) != 1:
                        keypoints = self.relaxedDetector.detect(preprocessorImage1)
                        keypointColor = (39, 127, 255)

                        if len(keypoints) != 1:
                            keypoints = self.superRelaxedDetector.detect(
                                preprocessorImage2
                            )
                            keypointColor = (39, 255, 127)
                            if len(keypoints) != 1:
                                keypoints = None
                            else:
                                self.__algorithm = 5
                        else:
                            self.__algorithm = 4
                    else:
                        self.__algorithm = 3
                else:
                    self.__algorithm = 2
            else:
                self.__algorithm = 1
        elif self.__algorithm == 1:
            preprocessorImage0 = self.preprocessImage(
                frameInput=nozzleDetectFrame, algorithm=0
            )
            keypoints = self.detector.detect(preprocessorImage0)
            keypointColor = (0, 0, 255)
        elif self.__algorithm == 2:
            preprocessorImage1 = self.preprocessImage(
                frameInput=nozzleDetectFrame, algorithm=1
            )
            keypoints = self.detector.detect(preprocessorImage1)
            keypointColor = (0, 255, 0)
        elif self.__algorithm == 3:
            preprocessorImage0 = self.preprocessImage(
                frameInput=nozzleDetectFrame, algorithm=0
            )
            keypoints = self.relaxedDetector.detect(preprocessorImage0)
            keypointColor = (255, 0, 0)
        else:
            preprocessorImage1 = self.preprocessImage(
                frameInput=nozzleDetectFrame, algorithm=1
            )
            keypoints = self.relaxedDetector.detect(preprocessorImage1)
            keypointColor = (39, 127, 255)

        if keypoints is not None:
            self.log(
                "Nozzle detected %i circles with algorithm: %s"
                % (len(keypoints), str(self.__algorithm))
            )
        else:
            self.log("Nozzle detection failed.")

        if keypoints is not None and len(keypoints) >= 1:
            if len(keypoints) > 1:
                closest_index = self.find_closest_keypoint(
                    keypoints, nozzleDetectFrame.shape
                )
                x, y = np.around(keypoints[closest_index].pt)
            else:
                x, y = np.around(keypoints[0].pt)

            x, y = int(x), int(y)
            center = (x, y)
            keypointRadius = int(np.around(keypoints[0].size / 2))
            displayFrame = self.draw_detection_overlay(
                displayFrame,
                center=center,
                keypoint_radius=keypointRadius,
                keypoint_color=keypointColor,
            )
        else:
            center = None
            displayFrame = self.draw_detection_overlay(
                displayFrame,
                center=None,
                keypoint_radius=17,
            )

        return center, displayFrame

    def preprocessImage(self, frameInput, algorithm=0):
        try:
            preprocess_mode = self.preprocess_mode
            if preprocess_mode == "off":
                return copy.deepcopy(frameInput)

            if preprocess_mode == "soft":
                gray = cv2.cvtColor(frameInput, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            gamma = max(0.1, self.gamma_value)
            outputFrame = self.adjust_gamma(image=frameInput, gamma=gamma)
        except Exception:
            outputFrame = copy.deepcopy(frameInput)

        if algorithm == 0:
            yuv = cv2.cvtColor(outputFrame, cv2.COLOR_BGR2YUV)
            yuvPlanes = cv2.split(yuv)
            yuvPlanes_0 = cv2.GaussianBlur(yuvPlanes[0], (7, 7), 6)
            yuvPlanes_0 = cv2.adaptiveThreshold(
                yuvPlanes_0,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                35,
                1,
            )
            outputFrame = cv2.cvtColor(yuvPlanes_0, cv2.COLOR_GRAY2BGR)
        elif algorithm == 1:
            outputFrame = cv2.cvtColor(outputFrame, cv2.COLOR_BGR2GRAY)
            _, outputFrame = cv2.threshold(
                outputFrame, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE
            )
            outputFrame = cv2.GaussianBlur(outputFrame, (7, 7), 6)
            outputFrame = cv2.cvtColor(outputFrame, cv2.COLOR_GRAY2BGR)
        elif algorithm == 2:
            gray = cv2.cvtColor(frameInput, cv2.COLOR_BGR2GRAY)
            outputFrame = cv2.medianBlur(gray, 5)

        return outputFrame

    def find_closest_keypoint(self, keypoints, frame_shape):
        height, width = frame_shape[:2]
        target_point = np.array([width // 2, height // 2])

        closest_index = None
        closest_distance = float("inf")

        for i, keypoint in enumerate(keypoints):
            point = np.array(keypoint.pt)
            distance = np.linalg.norm(point - target_point)

            if distance < closest_distance:
                closest_distance = distance
                closest_index = i

        return closest_index

    def adjust_gamma(self, image, gamma=1.2):
        invGamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image, table)
