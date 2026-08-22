import sys
print("Python:", sys.version.split()[0])
import flask
print("Flask:", flask.__version__ if hasattr(flask, "__version__") else "installed")
import numpy
print("NumPy:", numpy.__version__)
import cv2
print("OpenCV:", cv2.__version__)
if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
    raise SystemExit("ERROR: cv2.face missing. Install opencv-contrib-python.")
import firebase_admin
print("Firebase Admin: installed")
from app import app
for rule in sorted(str(r) for r in app.url_map.iter_rules()):
    pass
print("Flask routes: OK")
print("Self-check passed.")
