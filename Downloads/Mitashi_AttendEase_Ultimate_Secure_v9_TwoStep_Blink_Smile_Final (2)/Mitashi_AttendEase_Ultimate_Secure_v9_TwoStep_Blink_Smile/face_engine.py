from __future__ import annotations

import base64
import statistics
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np

_MODEL_CACHE: Dict[str, Tuple[str, object, Dict[int, str]]] = {}
_CACHE_LOCK = threading.RLock()
_CASCADE_CACHE: Dict[str, object] = {}
_CASCADE_LOCK = threading.RLock()


def _require_lbph() -> None:
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError("OpenCV face module is unavailable. Run setup_windows.bat to install opencv-contrib-python.")


def _cascade_named(filename: str):
    with _CASCADE_LOCK:
        cached = _CASCADE_CACHE.get(filename)
        if cached is not None:
            return cached
        path = Path(cv2.data.haarcascades) / filename
        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            raise RuntimeError(f"OpenCV cascade could not be loaded: {filename}")
        _CASCADE_CACHE[filename] = cascade
        return cascade


def _cascade():
    return _cascade_named("haarcascade_frontalface_default.xml")


def _eye_cascade():
    return _cascade_named("haarcascade_eye_tree_eyeglasses.xml")


def _smile_cascade():
    return _cascade_named("haarcascade_smile.xml")


def decode_data_url(data_url: str) -> bytes:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


def _decode_image(raw: bytes):
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Camera frame could not be decoded.")
    return image


def _largest_face(gray: np.ndarray):
    faces = _cascade().detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(90, 90))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def _raw_face_crop(raw: bytes) -> Optional[np.ndarray]:
    image = _decode_image(raw)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    box = _largest_face(gray)
    if box is None:
        return None
    x, y, w, h = box
    crop = gray[y:y + h, x:x + w]
    return cv2.resize(crop, (160, 160))


def _normalize_face(crop: np.ndarray) -> np.ndarray:
    # Normalize illumination before LBPH comparison. This is applied both to
    # stored enrollment samples at model-build time and to live verification
    # frames, so existing enrollments continue to work without re-enrollment.
    return cv2.equalizeHist(cv2.resize(crop, (160, 160)))


def face_crop(raw: bytes) -> Optional[np.ndarray]:
    crop = _raw_face_crop(raw)
    if crop is None:
        return None
    return _normalize_face(crop)


def crop_to_jpeg(raw: bytes, quality: int = 70) -> Optional[bytes]:
    # Keep the cloud enrollment sample neutral/raw. Normalization is applied
    # consistently when the recognition model is built.
    crop = _raw_face_crop(raw)
    if crop is None:
        return None
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Could not encode face crop.")
    return encoded.tobytes()


def invalidate_class_model(cache_key: str) -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.pop(cache_key, None)


def get_cached_model(cache_key: str, version: str):
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached and cached[0] == version:
            return cached[1], cached[2]
    return None


def build_or_get_model(cache_key: str, version: str, face_samples: Dict[str, List[bytes]]):
    _require_lbph()
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached and cached[0] == version:
            return cached[1], cached[2]

    images: List[np.ndarray] = []
    labels: List[int] = []
    label_map: Dict[int, str] = {}
    label = 0
    for student_id in sorted(face_samples):
        usable = 0
        for raw in face_samples[student_id]:
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            images.append(_normalize_face(image))
            labels.append(label)
            usable += 1
        if usable:
            label_map[label] = student_id
            label += 1

    if not images:
        raise RuntimeError("No enrolled Firebase face samples are available for this class.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=80.0)
    recognizer.train(images, np.asarray(labels, dtype=np.int32))
    with _CACHE_LOCK:
        _MODEL_CACHE[cache_key] = (version, recognizer, label_map)
    return recognizer, label_map


def identify_frames(
    raw_frames: Iterable[bytes],
    recognizer,
    label_map: Dict[int, str],
    allowed_student_ids: Optional[Set[str]] = None,
    threshold: float = 75.0,
) -> Tuple[Optional[str], float, int, int]:
    allowed = {x.upper() for x in allowed_student_ids} if allowed_student_ids else None
    matches: Dict[str, List[float]] = defaultdict(list)
    usable = 0
    best_any = 999.0

    for raw in raw_frames:
        crop = face_crop(raw)
        if crop is None:
            continue
        usable += 1
        label, distance = recognizer.predict(crop)
        distance = float(distance)
        best_any = min(best_any, distance)
        student_id = label_map.get(int(label))
        if not student_id:
            continue
        if allowed is not None and student_id.upper() not in allowed:
            continue
        if distance <= threshold:
            matches[student_id].append(distance)

    if usable == 0:
        return None, 999.0, 0, 0
    # Use multi-frame voting. With a longer 10-frame verification burst,
    # three agreeing frames is strong enough while still tolerating brief blur.
    required = 3 if usable >= 7 else 2
    if not matches:
        return None, best_any, usable, 0
    ranked = sorted(matches.items(), key=lambda item: (-len(item[1]), statistics.median(item[1])))
    student_id, distances = ranked[0]
    if len(distances) < required:
        return None, min(distances), usable, len(distances)
    return student_id, float(statistics.median(distances)), usable, len(distances)


# ---------------- Random challenge-response liveness ----------------
# This is a lightweight anti-spoofing layer for a classroom demo. It is intended
# to stop simple static-photo and ordinary pre-recorded-video replay attempts.
# It is not a replacement for a certified depth/IR anti-spoofing system.

def _face_observation(raw: bytes) -> Optional[Dict[str, float]]:
    image = _decode_image(raw)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ih, iw = gray.shape[:2]
    faces = _cascade().detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(90, 90))
    if len(faces) != 1:
        return None
    x, y, w, h = [int(v) for v in faces[0]]
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    upper = roi[:max(1, int(h * 0.62)), :]
    eyes = _eye_cascade().detectMultiScale(
        upper, scaleFactor=1.1, minNeighbors=6, minSize=(max(16, w // 10), max(12, h // 12))
    )

    lower_y = int(h * 0.42)
    lower = roi[lower_y:, :]
    smiles = _smile_cascade().detectMultiScale(
        lower, scaleFactor=1.65, minNeighbors=16, minSize=(max(24, w // 5), max(12, h // 10))
    )

    return {
        "center_x": (x + w / 2.0) / max(iw, 1),
        "center_y": (y + h / 2.0) / max(ih, 1),
        "area": (w * h) / float(max(iw * ih, 1)),
        "eyes": float(len(eyes)),
        "smile": 1.0 if len(smiles) > 0 else 0.0,
    }


def _observations(raw_frames: Iterable[bytes]) -> List[Dict[str, float]]:
    items: List[Dict[str, float]] = []
    for raw in raw_frames:
        try:
            obs = _face_observation(raw)
        except Exception:
            obs = None
        if obs is not None:
            items.append(obs)
    return items




def _region_motion(raw_frames: Iterable[bytes], y1: int, y2: int, x1: int, x2: int) -> float:
    rois: List[np.ndarray] = []
    for raw in raw_frames:
        try:
            crop = face_crop(raw)
        except Exception:
            crop = None
        if crop is None:
            continue
        roi = crop[y1:y2, x1:x2]
        if roi.size:
            rois.append(cv2.GaussianBlur(roi, (5, 5), 0))
    if len(rois) < 3:
        return 0.0
    diffs = [float(np.mean(cv2.absdiff(rois[i], rois[i - 1])) / 255.0) for i in range(1, len(rois))]
    if not diffs:
        return 0.0
    # A brief expression can be diluted by quiet frames, so use the stronger part
    # of the sequence rather than a simple global average.
    diffs = sorted(diffs, reverse=True)
    return float(statistics.mean(diffs[: min(3, len(diffs))]))

def liveness_baseline(raw_frames: Iterable[bytes]) -> Dict[str, float]:
    items = _observations(raw_frames)
    if len(items) < 3:
        raise ValueError("Keep one clear face centered in good lighting for calibration.")
    return {
        "center_x": float(statistics.median(x["center_x"] for x in items)),
        "center_y": float(statistics.median(x["center_y"] for x in items)),
        "area": float(statistics.median(x["area"] for x in items)),
        "eye_open_ratio": float(sum(1 for x in items if x["eyes"] >= 1.0) / len(items)),
        "usable": float(len(items)),
    }


def verify_liveness_action(action: str, raw_frames: Iterable[bytes], baseline: Dict[str, float]):
    raw_frames = list(raw_frames)
    items = _observations(raw_frames)
    if len(items) < 4:
        return False, "Face was not clear for enough frames.", {"usable": len(items)}

    centers = [x["center_x"] for x in items]
    areas = [x["area"] for x in items]
    eye_states = [x["eyes"] >= 1.0 for x in items]
    smile_ratio = sum(x["smile"] for x in items) / len(items)
    center_x = float(statistics.median(centers))
    area = float(statistics.median(areas))
    base_x = float(baseline.get("center_x", 0.5))
    base_area = max(float(baseline.get("area", 0.01)), 0.001)

    eye_motion = _region_motion(raw_frames, 32, 82, 18, 142)
    mouth_motion = _region_motion(raw_frames, 88, 145, 28, 132)
    diagnostics = {
        "usable": len(items),
        "center_x": round(center_x, 3),
        "baseline_x": round(base_x, 3),
        "area_ratio": round(area / base_area, 3),
        "smile_ratio": round(smile_ratio, 3),
        "eye_motion": round(eye_motion, 4),
        "mouth_motion": round(mouth_motion, 4),
        "eyes_seen": int(sum(1 for v in eye_states if v)),
        "eyes_missing": int(sum(1 for v in eye_states if not v)),
    }

    if action == "move_left":
        # Do not use the median of the whole sequence: the first frames are
        # naturally still near center while the student begins moving. Detect
        # a clear directional excursion and require it in multiple frames.
        left_peak = min(centers)
        left_hits = sum(1 for x in centers if x <= base_x - 0.025)
        right_peak = max(centers)
        diagnostics.update({
            "left_peak": round(left_peak, 3),
            "left_delta": round(base_x - left_peak, 3),
            "left_hits": left_hits,
        })
        passed = (base_x - left_peak) >= 0.040 and left_hits >= 2
        if passed:
            return True, "Left movement detected.", diagnostics
        if (right_peak - base_x) >= 0.040:
            return False, "Movement was detected toward the RIGHT. Move toward the LEFT side of the frame.", diagnostics
        return False, "Move your whole face farther toward the LEFT side of the frame and hold briefly.", diagnostics

    if action == "move_right":
        right_peak = max(centers)
        right_hits = sum(1 for x in centers if x >= base_x + 0.025)
        left_peak = min(centers)
        diagnostics.update({
            "right_peak": round(right_peak, 3),
            "right_delta": round(right_peak - base_x, 3),
            "right_hits": right_hits,
        })
        passed = (right_peak - base_x) >= 0.040 and right_hits >= 2
        if passed:
            return True, "Right movement detected.", diagnostics
        if (base_x - left_peak) >= 0.040:
            return False, "Movement was detected toward the LEFT. Move toward the RIGHT side of the frame.", diagnostics
        return False, "Move your whole face farther toward the RIGHT side of the frame and hold briefly.", diagnostics

    if action == "move_closer":
        # Same idea for distance: use the strongest stable approach rather than
        # the median, because the sequence begins at the calibrated distance.
        peak_area = max(areas)
        close_hits = sum(1 for x in areas if x >= base_area * 1.08)
        diagnostics.update({"peak_area_ratio": round(peak_area / base_area, 3), "close_hits": close_hits})
        passed = peak_area >= base_area * 1.12 and close_hits >= 2
        return passed, "Move a little closer to the camera and hold." if not passed else "Closer movement detected.", diagnostics

    if action == "smile":
        # Require temporal mouth-region change as well as (or stronger than) the
        # Haar smile cue. This prevents a static smiling photo from passing simply
        # because it already contains a smile.
        passed = mouth_motion >= 0.016 and (smile_ratio >= 0.15 or mouth_motion >= 0.028)
        return passed, "Smile clearly, then relax your mouth slightly." if not passed else "Live mouth movement detected.", diagnostics

    if action == "blink":
        # Haar eye detection is deliberately evaluated over a sequence instead of a
        # single frame. A blink should contain both eye-visible and eye-missing frames.
        open_count = sum(1 for v in eye_states if v)
        closed_count = len(eye_states) - open_count
        transition = any(eye_states[i] != eye_states[i - 1] for i in range(1, len(eye_states)))
        passed = (open_count >= 2 and closed_count >= 1 and transition) or eye_motion >= 0.018
        return passed, "Blink naturally two times while keeping your face centered." if not passed else "Live eye movement detected.", diagnostics

    return False, "Unknown liveness action.", diagnostics
