import cv2
import numpy as np
import copy


class KTAMV_DM:

    def __init__(self):
        self.detector = self.create_blob_detector()

    def create_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 5000

        params.filterByCircularity = True
        params.minCircularity = 0.5

        return cv2.SimpleBlobDetector_create(params)

    # 🔥 DYNAMISCHE MITTE
    def get_center(self, frame):
        h, w = frame.shape[:2]
        return w // 2, h // 2, w, h

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def find_closest_keypoint(self, keypoints, target):
        closest = None
        closest_dist = float("inf")

        for kp in keypoints:
            pt = np.array(kp.pt)
            dist = np.linalg.norm(pt - target)

            if dist < closest_dist:
                closest_dist = dist
                closest = kp

        return closest

    def draw_overlay(self, frame, center=None, radius=20):
        cx, cy, w, h = self.get_center(frame)

        # 🔥 FADENKREUZ
        frame = cv2.line(frame, (cx, 0), (cx, h), (0, 0, 0), 2)
        frame = cv2.line(frame, (0, cy), (w, cy), (0, 0, 0), 2)
        frame = cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)
        frame = cv2.line(frame, (0, cy), (w, cy), (255, 255, 255), 1)

        if center is None:
            return frame

        px, py = int(center[0]), int(center[1])

        # 🔥 DÜSEN-MARKER
        frame = cv2.circle(frame, (px, py), radius, (0, 0, 0), 3)
        frame = cv2.circle(frame, (px, py), radius + 1, (0, 255, 0), 1)

        # kleines Kreuz auf Düse
        frame = cv2.line(frame, (px - 10, py), (px + 10, py), (255, 255, 255), 1)
        frame = cv2.line(frame, (px, py - 10), (px, py + 10), (255, 255, 255), 1)

        return frame

    def nozzleDetection(self, frame):
        work = copy.deepcopy(frame)

        processed = self.preprocess(work)

        keypoints = self.detector.detect(processed)

        cx, cy, _, _ = self.get_center(work)
        target = np.array([cx, cy])

        if not keypoints:
            return None, self.draw_overlay(work, None)

        kp = self.find_closest_keypoint(keypoints, target)

        if kp is None:
            return None, self.draw_overlay(work, None)

        center = (int(kp.pt[0]), int(kp.pt[1]))
        radius = max(6, int(kp.size / 2))

        return center, self.draw_overlay(work, center, radius)
