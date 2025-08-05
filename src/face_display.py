import cv2
import numpy as np
import asyncio

DEBUG = True
# -----------------------------------------------------------------------------
# Class to display and animate a symbolic face
# -----------------------------------------------------------------------------
class FaceDisplay:
    def __init__(self, window_name="Assistant Face"):
        self.window_name = window_name
        self.face_size = (300, 300)
        self.background_color = (20, 20, 20)
        self.foreground_color = (255, 255, 255)
        self.faces = {}
        self._generate_faces()
        cv2.namedWindow(self.window_name)

    def _generate_faces(self):
        if DEBUG: print("\n=== Generating symbolic face images ===")
        font = cv2.FONT_HERSHEY_SIMPLEX
        eye_y = 130
        
        # --- Waiting / Speaking (mouth closed) face ---
        face_wait = np.full((*self.face_size, 3), self.background_color, dtype=np.uint8)
        cv2.circle(face_wait, (100, eye_y), 10, self.foreground_color, -1)
        cv2.circle(face_wait, (200, eye_y), 10, self.foreground_color, -1)
        
        mouth_text = "____"
        mouth_font_scale, mouth_thickness = 1.2, 8
        (text_w, _), _ = cv2.getTextSize(mouth_text, font, mouth_font_scale, mouth_thickness)
        mouth_x = (self.face_size[0] - text_w) // 2
        cv2.putText(face_wait, mouth_text, (mouth_x, 220), font, mouth_font_scale, self.foreground_color, mouth_thickness)
        self.faces["waiting"] = face_wait
        self.faces["speaking_closed"] = face_wait

        # --- Scanning (left) face ---
        face_scan_l = np.full((*self.face_size, 3), self.background_color, dtype=np.uint8)
        cv2.circle(face_scan_l, (80, eye_y), 10, self.foreground_color, -1)
        cv2.circle(face_scan_l, (180, eye_y), 10, self.foreground_color, -1)
        cv2.putText(face_scan_l, mouth_text, (mouth_x, 220), font, mouth_font_scale, self.foreground_color, mouth_thickness)
        self.faces["scanning_left"] = face_scan_l

        # --- Scanning (right) face ---
        face_scan_r = np.full((*self.face_size, 3), self.background_color, dtype=np.uint8)
        cv2.circle(face_scan_r, (120, eye_y), 10, self.foreground_color, -1)
        cv2.circle(face_scan_r, (220, eye_y), 10, self.foreground_color, -1)
        cv2.putText(face_scan_r, mouth_text, (mouth_x, 220), font, mouth_font_scale, self.foreground_color, mouth_thickness)
        self.faces["scanning_right"] = face_scan_r

        # --- Speaking (mouth open) face ---
        face_open = np.full((*self.face_size, 3), self.background_color, dtype=np.uint8)
        cv2.circle(face_open, (100, eye_y), 10, self.foreground_color, -1)
        cv2.circle(face_open, (200, eye_y), 10, self.foreground_color, -1)
        cv2.ellipse(face_open, (150, 210), (40, 25), 0, 0, 360, self.foreground_color, -1)
        self.faces["speaking_open"] = face_open
        if DEBUG: print("  ✓ Face image generation complete")

    def show(self, face_key):
        if face_key in self.faces:
            cv2.imshow(self.window_name, self.faces[face_key])
            cv2.waitKey(1)

    async def animate_scanning(self):
        if DEBUG: print("  [▶] Starting scanning animation")
        is_left = True
        try:
            while True:
                self.show("scanning_left" if is_left else "scanning_right")
                is_left = not is_left
                await asyncio.sleep(0.4)
        except asyncio.CancelledError:
            if DEBUG: print("  [■] Stopping scanning animation")
            self.show("waiting")

    async def animate_speaking(self, duration):
        if DEBUG: print("  [▶] Starting speaking animation")
        start_time = time.time()
        is_open = True
        while time.time() - start_time < duration:
            self.show("speaking_open" if is_open else "speaking_closed")
            is_open = not is_open
            await asyncio.sleep(0.15)
        self.show("waiting")
        if DEBUG: print("  [■] Speaking animation finished")
