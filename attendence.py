import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import csv
import pandas as pd
from datetime import datetime

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "1234"
FACES_DIR = "faces"
ATTENDANCE_FILE = "attendance.csv"

def authenticate():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")

        if login_btn:
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.success("✅ Authentication successful!")
            else:
                st.error("❌ Invalid credentials")

    return st.session_state.authenticated

def load_known_faces():
    encodings, names, roles = [], [], []
    for role in ["students", "teachers"]:
        role_path = os.path.join(FACES_DIR, role)
        if not os.path.exists(role_path):
            continue
        for file in os.listdir(role_path):
            img_path = os.path.join(role_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb)
            if not face_locations:
                continue
            face_enc = face_recognition.face_encodings(rgb, face_locations)[0]
            encodings.append(face_enc)
            names.append(os.path.splitext(file)[0].upper())
            roles.append(role[:-1])
    return encodings, names, roles

def mark_attendance(name, role, session):
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")
    file_exists = os.path.exists(ATTENDANCE_FILE)

    if file_exists:
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Role", "Date", "Time", "Session"])

    if not ((df["Name"] == name) & (df["Date"] == date) & (df["Session"] == session)).any():
        new_entry = pd.DataFrame([[name, role, date, time, session]],
                                 columns=["Name", "Role", "Date", "Time", "Session"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        st.success(f"✅ Marked {name} ({role}) for {session}")

def add_user():
    st.subheader("➕ Add New User (Admin Only)")
    if authenticate():
        role = st.selectbox("Select Role", ["student", "teacher"])
        name = st.text_input("Enter Name").strip().upper()
        uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
        save_btn = st.button("Save User")

        if save_btn:
            if not name:
                st.error("❌ Please enter a name.")
            elif not uploaded_file:
                st.error("❌ Please upload a face image.")
            else:
                role_folder = os.path.join(FACES_DIR, f"{role}s")
                os.makedirs(role_folder, exist_ok=True)
                img_path = os.path.join(role_folder, f"{name}.jpg")
                with open(img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ {role.title()} '{name}' added successfully!")

def view_records():
    st.subheader("📑 Attendance Records (Admin Only)")
    if authenticate():
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_csv(ATTENDANCE_FILE)
            st.dataframe(df)
        else:
            st.warning("⚠️ No attendance records found.")

def attendance():
    st.subheader("📸 Mark Attendance")
    session_choice = st.radio("Select Session", ["Morning", "Evening"])
    start_cam = st.button("Start Camera")

    known_encodings, known_names, known_roles = load_known_faces()
    if not known_encodings:
        st.error("❌ No valid faces found. Please add users first.")
        return

    FRAME_WINDOW = st.image([])

    if start_cam:
        cap = cv2.VideoCapture(0)
        stop_cam = st.button("Stop Camera")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture frame")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, locations)

            for encoding, loc in zip(encodings, locations):
                matches = face_recognition.compare_faces(known_encodings, encoding)
                face_distances = face_recognition.face_distance(known_encodings, encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    role = known_roles[best_match_index]
                    y1, x2, y2, x1 = loc
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({role})", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    mark_attendance(name, role, session_choice)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if stop_cam:
                break

        cap.release()

def main():
    st.set_page_config(page_title="Smart Attendance System", layout="wide")
    st.title("🎓 Smart Attendance System")

    menu = st.sidebar.radio("Menu", ["📸 Mark Attendance", "➕ Add User", "📑 View Records"])

    if menu == "📸 Mark Attendance":
        attendance()
    elif menu == "➕ Add User":
        add_user()
    elif menu == "📑 View Records":
        view_records()

if __name__ == "__main__":
    main()
