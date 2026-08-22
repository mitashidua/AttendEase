from __future__ import annotations

import base64
import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "firebase_config.json"
SERVICE_ACCOUNT_FILE = CONFIG_DIR / "firebase_service_account.json"
ROOT = "attendease"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class FirebaseCloudStore:
    """Firebase Realtime Database is the primary persistent store.

    The app stores teacher accounts, teacher-specific class buckets, rosters,
    one-time enrolled face samples, attendance sessions and attendance records
    in Firebase Realtime Database. The classroom computer keeps no persistent
    student face dataset or trained recognizer.

    Only the Firebase connection configuration and service-account credential
    are stored locally so that this server can connect to the user's project.
    """

    def __init__(self) -> None:
        self.app = None
        self.error = ""
        self.database_url = ""
        self.initialize()

    def configured(self) -> bool:
        return CONFIG_FILE.exists() and SERVICE_ACCOUNT_FILE.exists()

    def initialize(self) -> bool:
        self.app = None
        self.error = ""
        if not self.configured():
            return False
        try:
            config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            self.database_url = str(config.get("database_url", "")).strip().rstrip("/")
            if not self.database_url.startswith("https://"):
                raise ValueError("Firebase Realtime Database URL is missing or invalid.")

            import firebase_admin
            from firebase_admin import credentials

            try:
                self.app = firebase_admin.get_app("attendease-rtdb")
            except ValueError:
                cred = credentials.Certificate(str(SERVICE_ACCOUNT_FILE))
                self.app = firebase_admin.initialize_app(
                    cred,
                    {"databaseURL": self.database_url},
                    name="attendease-rtdb",
                )
            self._seed_demo_teacher_if_empty()
            return True
        except Exception as exc:
            self.error = str(exc)
            self.app = None
            return False

    def save_configuration(self, database_url: str, service_account_bytes: bytes) -> None:
        database_url = database_url.strip().rstrip("/")
        if not database_url.startswith("https://"):
            raise ValueError("Enter the HTTPS Realtime Database URL from Firebase Console.")
        parsed = json.loads(service_account_bytes.decode("utf-8"))
        if parsed.get("type") != "service_account":
            raise ValueError("The uploaded JSON is not a Firebase service-account key.")
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        SERVICE_ACCOUNT_FILE.write_bytes(service_account_bytes)
        CONFIG_FILE.write_text(json.dumps({"database_url": database_url}, indent=2), encoding="utf-8")
        if not self.initialize():
            raise ValueError(self.error or "Firebase connection failed.")

    def status(self) -> Tuple[bool, str]:
        if self.app is not None:
            return True, "Firebase Realtime Database connected"
        if self.configured() and self.error:
            return False, f"Firebase configuration error: {self.error}"
        return False, "Firebase setup required"

    def _db(self, path: str):
        if self.app is None:
            raise RuntimeError("Firebase is not configured yet.")
        from firebase_admin import db
        return db.reference(path, app=self.app)

    # ---------- teacher accounts ----------
    def _seed_demo_teacher_if_empty(self) -> None:
        teachers = self._db(f"{ROOT}/teachers").get() or {}
        if teachers:
            return
        tid = "teacher_demo"
        self._db(f"{ROOT}/teachers/{tid}/profile").set({
            "teacher_id": tid,
            "name": "Faculty Admin",
            "email": "teacher@attendease.local",
            "email_lower": "teacher@attendease.local",
            "password_hash": generate_password_hash("Teacher@123"),
            "created_at": iso_now(),
        })

    def list_teachers(self) -> Dict[str, Dict[str, Any]]:
        raw = self._db(f"{ROOT}/teachers").get() or {}
        out: Dict[str, Dict[str, Any]] = {}
        for tid, node in raw.items():
            profile = (node or {}).get("profile") or {}
            if profile:
                out[tid] = profile
        return out

    def create_teacher(self, name: str, email: str, password: str) -> Dict[str, Any]:
        name = " ".join(name.strip().split())
        email = email.strip().lower()
        if not name or not email or len(password) < 6:
            raise ValueError("Name, email, and a password of at least 6 characters are required.")
        for teacher in self.list_teachers().values():
            if teacher.get("email_lower") == email:
                raise ValueError("A teacher account with this email already exists.")
        tid = "tch_" + secrets.token_hex(6)
        profile = {
            "teacher_id": tid,
            "name": name,
            "email": email,
            "email_lower": email,
            "password_hash": generate_password_hash(password),
            "created_at": iso_now(),
        }
        self._db(f"{ROOT}/teachers/{tid}/profile").set(profile)
        return profile

    def authenticate_teacher(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        email = email.strip().lower()
        for profile in self.list_teachers().values():
            if profile.get("email_lower") == email and check_password_hash(profile.get("password_hash", ""), password):
                return profile
        return None

    def get_teacher(self, teacher_id: str) -> Optional[Dict[str, Any]]:
        return self._db(f"{ROOT}/teachers/{teacher_id}/profile").get()

    # ---------- classes ----------
    def _classes_ref(self, teacher_id: str):
        return self._db(f"{ROOT}/teachers/{teacher_id}/classes")

    def list_classes(self, teacher_id: str) -> List[Dict[str, Any]]:
        raw = self._classes_ref(teacher_id).get() or {}
        classes: List[Dict[str, Any]] = []
        for class_id, node in raw.items():
            meta = (node or {}).get("meta") or {}
            if meta:
                meta.setdefault("class_id", class_id)
                classes.append(meta)
        return sorted(classes, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_class(self, teacher_id: str, class_id: str) -> Optional[Dict[str, Any]]:
        return self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}/meta").get()

    def create_class(self, teacher_id: str, name: str, section: str = "") -> Dict[str, Any]:
        name = " ".join(name.strip().split())
        section = " ".join(section.strip().split())
        if not name:
            raise ValueError("Class / subject name is required.")
        class_id = "cls_" + secrets.token_hex(5)
        code = "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        meta = {
            "class_id": class_id,
            "code": code,
            "name": name,
            "section": section,
            "teacher_id": teacher_id,
            "created_at": iso_now(),
        }
        self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}/meta").set(meta)
        return meta

    def delete_class(self, teacher_id: str, class_id: str) -> None:
        self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}").delete()

    # ---------- roster ----------
    def list_students(self, teacher_id: str, class_id: str) -> List[Dict[str, Any]]:
        raw = self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}/students").get() or {}
        return sorted(raw.values(), key=lambda s: s.get("student_id", ""))

    def get_student(self, teacher_id: str, class_id: str, student_id: str) -> Optional[Dict[str, Any]]:
        return self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}/students/{student_id}").get()

    def add_student(self, teacher_id: str, class_id: str, name: str, student_id: str) -> Dict[str, Any]:
        if not self.get_class(teacher_id, class_id):
            raise ValueError("Class not found.")
        name = " ".join(name.strip().split())
        student_id = re.sub(r"[.#$\[\]/]", "-", student_id.strip().upper())
        if not name or not student_id:
            raise ValueError("Student name and student ID are required.")
        if self.get_student(teacher_id, class_id, student_id):
            raise ValueError("This student ID already exists in the class.")
        student = {
            "student_id": student_id,
            "name": name,
            "face_enrolled": False,
            "sample_count": 0,
            "created_at": iso_now(),
            "updated_at": iso_now(),
        }
        self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}/students/{student_id}").set(student)
        return student

    def update_student_face_state(self, teacher_id: str, class_id: str, student_id: str, sample_count: int, enrolled: bool) -> None:
        self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}/students/{student_id}").update({
            "sample_count": int(sample_count),
            "face_enrolled": bool(enrolled),
            "updated_at": iso_now(),
        })

    # ---------- one-time face enrollment in Realtime Database ----------
    def _face_ref(self, teacher_id: str, class_id: str, student_id: str):
        return self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}/face_templates/{student_id}")

    def reset_face_samples(self, teacher_id: str, class_id: str, student_id: str) -> None:
        self._face_ref(teacher_id, class_id, student_id).delete()
        self.update_student_face_state(teacher_id, class_id, student_id, 0, False)

    def save_face_sample(self, teacher_id: str, class_id: str, student_id: str, sample_no: int, jpeg_bytes: bytes) -> str:
        """Store a normalized, compressed face crop as base64 in RTDB.

        The browser frame is never saved. Only the detected/normalized grayscale face
        crop is encoded. This remains biometric data and should be protected accordingly.
        """
        encoded = base64.b64encode(jpeg_bytes).decode("ascii")
        key = f"sample_{sample_no:02d}"
        self._face_ref(teacher_id, class_id, student_id).child(key).set(encoded)
        return key

    def download_class_faces(self, teacher_id: str, class_id: str) -> Dict[str, List[bytes]]:
        raw = self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}/face_templates").get() or {}
        result: Dict[str, List[bytes]] = {}
        enrolled = {s["student_id"] for s in self.list_students(teacher_id, class_id) if s.get("face_enrolled")}
        for student_id, samples_node in raw.items():
            if student_id not in enrolled or not isinstance(samples_node, dict):
                continue
            samples: List[bytes] = []
            for key in sorted(samples_node):
                value = samples_node.get(key)
                if isinstance(value, str) and value:
                    try:
                        samples.append(base64.b64decode(value))
                    except Exception:
                        continue
            if samples:
                result[student_id] = samples
        return result

    def class_model_version(self, teacher_id: str, class_id: str) -> str:
        students = self.list_students(teacher_id, class_id)
        return "|".join(
            f"{s.get('student_id')}:{s.get('sample_count', 0)}:{s.get('updated_at', '')}"
            for s in students
        )

    # ---------- sessions / attendance ----------
    def list_sessions(self, teacher_id: str, class_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sessions: List[Dict[str, Any]] = []
        classes = [self.get_class(teacher_id, class_id)] if class_id else self.list_classes(teacher_id)
        for clazz in classes:
            if not clazz:
                continue
            cid = clazz["class_id"]
            raw = self._db(f"{ROOT}/teachers/{teacher_id}/classes/{cid}/sessions").get() or {}
            for sid, item in raw.items():
                item = item or {}
                item["session_id"] = sid
                item["class_id"] = cid
                if item.get("status") == "active" and parse_iso(item["ends_at"]) <= utc_now():
                    item["status"] = "ended"
                    item["ended_reason"] = "expired"
                    self._db(f"{ROOT}/teachers/{teacher_id}/classes/{cid}/sessions/{sid}").update({
                        "status": "ended",
                        "ended_reason": "expired",
                    })
                sessions.append(item)
        return sorted(sessions, key=lambda s: s.get("starts_at", ""), reverse=True)

    def get_session(self, teacher_id: str, class_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        item = self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}/sessions/{session_id}").get()
        if item:
            item["session_id"] = session_id
            item["class_id"] = class_id
        return item

    def find_session(self, teacher_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        for item in self.list_sessions(teacher_id):
            if item.get("session_id") == session_id:
                return item
        return None

    def session_active(self, attendance_session: Dict[str, Any]) -> bool:
        return bool(
            attendance_session
            and attendance_session.get("status") == "active"
            and parse_iso(attendance_session["ends_at"]) > utc_now()
        )

    def start_session(self, teacher_id: str, class_id: str, duration_minutes: int = 20) -> Dict[str, Any]:
        clazz = self.get_class(teacher_id, class_id)
        if not clazz:
            raise ValueError("Class not found.")
        if not any(s.get("face_enrolled") for s in self.list_students(teacher_id, class_id)):
            raise ValueError("Enroll at least one student face before starting attendance.")
        for item in self.list_sessions(teacher_id, class_id):
            if self.session_active(item):
                raise ValueError("This class already has an active attendance session.")
        duration_minutes = max(1, min(int(duration_minutes), 180))
        sid = "ses_" + secrets.token_hex(6)
        start = utc_now()
        item = {
            "session_id": sid,
            "class_id": class_id,
            "teacher_id": teacher_id,
            "status": "active",
            "starts_at": start.isoformat(),
            "ends_at": (start + timedelta(minutes=duration_minutes)).isoformat(),
            "duration_minutes": duration_minutes,
            "created_at": iso_now(),
        }
        self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}/sessions/{sid}").set(item)
        return item

    def stop_session(self, teacher_id: str, class_id: str, session_id: str) -> None:
        attendance_session = self.get_session(teacher_id, class_id, session_id)
        if not attendance_session:
            raise ValueError("Session not found.")
        self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}/sessions/{session_id}").update({
            "status": "ended",
            "ended_at": iso_now(),
            "ended_reason": "teacher",
        })

    def attendance_for_session(self, teacher_id: str, class_id: str, session_id: str) -> List[Dict[str, Any]]:
        raw = self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}/attendance/{session_id}").get() or {}
        return sorted(raw.values(), key=lambda r: r.get("marked_at", ""), reverse=True)

    def attendance_count_for_teacher(self, teacher_id: str) -> int:
        total = 0
        for clazz in self.list_classes(teacher_id):
            cid = clazz["class_id"]
            raw = self._db(f"{ROOT}/teachers/{teacher_id}/classes/{cid}/attendance").get() or {}
            total += sum(len(value or {}) for value in raw.values())
        return total

    def mark_attendance(self, teacher_id: str, class_id: str, session_id: str, student_id: str, face_distance: float, liveness_actions: Optional[List[str]] = None) -> Dict[str, Any]:
        attendance_session = self.get_session(teacher_id, class_id, session_id)
        if not attendance_session or not self.session_active(attendance_session):
            raise ValueError("Attendance session is closed.")
        student = self.get_student(teacher_id, class_id, student_id)
        if not student or not student.get("face_enrolled"):
            raise ValueError("Recognized student is not enrolled for this class.")
        ref = self._db(f"{ROOT}/teachers/{teacher_id}/classes/{class_id}/attendance/{session_id}/{student_id}")
        if ref.get():
            raise ValueError("Attendance already marked for this student in this session.")
        record = {
            "student_id": student_id,
            "student_name": student.get("name", ""),
            "class_id": class_id,
            "session_id": session_id,
            "status": "Present",
            "marked_at": iso_now(),
            "face_distance": round(float(face_distance), 2),
            "liveness_verified": bool(liveness_actions),
            "liveness_actions": list(liveness_actions or []),
        }
        ref.set(record)
        return record
