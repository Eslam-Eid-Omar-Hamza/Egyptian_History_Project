import sys
import time
import csv
import json
from pathlib import Path
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QComboBox, QMessageBox,
    QTextEdit, QFileDialog, QMainWindow, QAction, QStatusBar
)
import cv2

# --- YOLO / OpenCV libs ---
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "Assets"
MODELS_DIR = BASE_DIR / "Models"
DATA_DIR   = BASE_DIR / "Data"

# ------------------ CONFIG ------------------
PHARAOH_LEFT_PNG  = str(ASSETS_DIR / "Pharaoh_Small1.png")
PHARAOH_RIGHT_PNG = str(ASSETS_DIR / "Pharaoh_Small2.png")
MODEL_PATHS = {
    "Egyptian History": str(MODELS_DIR / "Egyptia_History.pt"),
    "GizaVision":       str(MODELS_DIR / "The_best_of_Giza.pt"),
}

EN_JSON_PATH = str(DATA_DIR / "StoryEN.json")
AR_JSON_PATH = str(DATA_DIR / "StoryAR.json")
HISTORY_FILE = str(BASE_DIR / "user_history.csv")



def get_history(cls_name):
    global story_data   # ← ضيف السطر ده
    for item in story_data:
        if item["name"].lower() == cls_name.lower():
            return f"{item['description']} (Year: {item['year_built']})\n\n{item['story']}"
    return "No history available."


# ---------- النصوص المتعددة اللغات ----------
texts = {
    "English": {
        "welcome": "Welcome {} to the Egyptian History Explorer App",
        "upload": "🖼 Load Image",
        "analyze": "🔍 Analyze Media",
        "delete": "🗑 Delete Image",
        "video": "🎬 Load Video",
        "camera": "📷 Open Camera",
        "back": "🔙 Back",
        "no_image": "No image loaded",
        "notes": "Welcome to Egyptian History",
        "created": "Created by: Eslam Eid Omar Hamza",
        "read_done": "Well done, {}! Landmark recorded.",
        "select_model": "Select Model:",
        "snapshot": "📸 Snapshot",
        "zoom_text": "🔍 Zoom Text"

    },
    "العربية": {
        "welcome": "أهلاً {} في تطبيق استكشاف التاريخ المصري",
        "upload": "🖼 تحميل صورة",
        "analyze": "🔍 تحليل الوسائط",
        "delete": "🗑 حذف الصورة",
        "video": "🎬 تحميل فيديو",
        "camera": "📷 فتح الكاميرا",
        "back": "🔙 رجوع",
        "no_image": "لم يتم تحميل صورة",
        "notes": "اهلا بك في التاريخ المصري العظيم:",
        "created": "صنع بواسطة: إسلام حمزة",
        "read_done": "برافو {}! تم تسجيل المعلم.",
        "select_model": "اختر الموديل:",
        "snapshot": "📸 لقطة",
        "zoom_text": "🔍 تكبير النص"
    }
}

# ------------------------ WELCOME ------------------------
class WelcomeWindow(QWidget):
    def __init__(self, start_callback):
        super().__init__()
        self.start_callback = start_callback
        self.setWindowTitle("Welcome — Egyptian History Explorer App")
        self.setFixedSize(900, 520)
        self._apply_style()
        layout = QVBoxLayout()
        
        title = QLabel("👑 Welcome to Egyptian History Explorer App 👑")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #FFD700; font-size: 26px; font-weight: bold;")
        layout.addWidget(title)

        row = QHBoxLayout()
        self.left_img = QLabel()
        pix_left = QPixmap(PHARAOH_LEFT_PNG)
        if not pix_left.isNull():
            self.left_img.setPixmap(pix_left.scaled(180, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        row.addWidget(self.left_img, alignment=Qt.AlignLeft)

        form = QVBoxLayout()
        lbl_name = QLabel("Name / الاسم:")
        lbl_name.setStyleSheet("color: #FFD700; font-weight:bold;")
        form.addWidget(lbl_name)
        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("Enter your name")
        self.input_name.setFixedWidth(340)
        self.input_name.setStyleSheet("padding:8px; background: white; color: black; border-radius:6px;")
        form.addWidget(self.input_name)

        lbl_lang = QLabel("Language / اللغة:")
        lbl_lang.setStyleSheet("color: #FFD700; font-weight:bold;")
        form.addWidget(lbl_lang)
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(["English", "العربية"])
        self.combo_lang.setFixedWidth(340)
        self.combo_lang.setStyleSheet("padding:6px; border-radius:6px; background: white; color: black;")
        form.addWidget(self.combo_lang)

        lbl_model = QLabel(texts["English"]["select_model"])
        lbl_model.setStyleSheet("color: #FFD700; font-weight:bold;")
        form.addWidget(lbl_model)
        self.combo_model = QComboBox()
        self.combo_model.addItems(list(MODEL_PATHS.keys()))
        self.combo_model.setFixedWidth(340)
        self.combo_model.setStyleSheet("padding:6px; border-radius:6px; background: white; color: black;")
        form.addWidget(self.combo_model)

        form.addSpacing(12)
        self.btn_start = QPushButton("▶ Start / ابدأ")
        self.btn_start.setFixedWidth(220)
        self.btn_start.setStyleSheet(self._gold_button_style())
        self.btn_start.clicked.connect(self.on_start)
        form.addWidget(self.btn_start, alignment=Qt.AlignCenter)

        row.addLayout(form)

        self.right_img = QLabel()
        pix_right = QPixmap(PHARAOH_RIGHT_PNG)
        if not pix_right.isNull():
            self.right_img.setPixmap(pix_right.scaled(180, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        row.addWidget(self.right_img, alignment=Qt.AlignRight)

        layout.addLayout(row)
        layout.addStretch()
        self.setLayout(layout)

    def _apply_style(self):
            pal = self.palette()
            pal.setColor(QPalette.Window, QColor("#111111"))
            self.setPalette(pal)

    def _gold_button_style(self):
        return ("QPushButton { background-color: #FFD700; color: black; font-weight: bold; "
                "border-radius: 10px; padding: 8px; } QPushButton:hover { background-color: #E6C200; }")

    def on_start(self):
        name = self.input_name.text().strip()
        lang = self.combo_lang.currentText()
        model_name = self.combo_model.currentText()
        if not name:
            QMessageBox.warning(self, "Missing name", "Please enter your name / الرجاء إدخال الاسم")
            return
        self.start_callback(name, lang, model_name, self)

# ------------------------ MAIN ------------------------
class MainWindow(QMainWindow):
    def __init__(self, user_name, language, model_name, welcome_window):
        super().__init__()
        self.user_name = user_name
        self.language = language
        self.model_name = model_name
        self.welcome_window = welcome_window
        self.setWindowTitle("Egypt History Explorer App")
        self.resize(1200, 780)
        self._apply_style()
        self.current_image_path = None
        self.notes_store = {}
        self.model = None
        self.cap = None
        self.timer = None

        central = QWidget()
        layout = QVBoxLayout(central)

        self.lbl_welcome = QLabel()
        self.lbl_welcome.setAlignment(Qt.AlignCenter)
        self.lbl_welcome.setStyleSheet("color: #FFD700; font-size: 22px; font-weight: bold;")
        layout.addWidget(self.lbl_welcome)

        controls = QHBoxLayout()
        self.btn_load = QPushButton()
        self.btn_load.setStyleSheet(self._gold_button_style())
        self.btn_load.clicked.connect(self.load_image)
        controls.addWidget(self.btn_load)

        self.btn_analyze = QPushButton()
        self.btn_analyze.setStyleSheet(self._gold_button_style())
        self.btn_analyze.clicked.connect(self.analyze_media)
        controls.addWidget(self.btn_analyze)

        self.btn_delete = QPushButton()
        self.btn_delete.setStyleSheet(self._gold_button_style())
        self.btn_delete.clicked.connect(self.delete_image)
        controls.addWidget(self.btn_delete)

        self.btn_resize = QPushButton("Full/Normal")
        self.btn_resize.setStyleSheet(self._gold_button_style())
        self.btn_resize.clicked.connect(self.toggle_window_size)
        controls.addWidget(self.btn_resize)

        self.btn_video = QPushButton()
        self.btn_video.setStyleSheet(self._gold_button_style())
        self.btn_video.clicked.connect(self.load_video)
        controls.addWidget(self.btn_video)

        self.btn_camera = QPushButton()
        self.btn_camera.setStyleSheet(self._gold_button_style())
        self.btn_camera.clicked.connect(self.open_camera)
        controls.addWidget(self.btn_camera)

        self.btn_snapshot = QPushButton()
        self.btn_snapshot.setStyleSheet(self._gold_button_style())
        self.btn_snapshot.clicked.connect(self.take_snapshot)
        controls.addWidget(self.btn_snapshot)
        self.btn_back = QPushButton()
        self.btn_back.setStyleSheet(self._gold_button_style())
        self.btn_back.clicked.connect(self.go_back)
        controls.addWidget(self.btn_back)

        controls.addStretch()
        layout.addLayout(controls)

        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setFixedSize(800, 500)
        self.lbl_image.setStyleSheet("background: #FFF8E1; border: 3px solid gold;")
        layout.addWidget(self.lbl_image, alignment=Qt.AlignCenter)

        btn_fullscreen = QPushButton("🔍 Zoom Text")
        btn_fullscreen.setStyleSheet(self._gold_button_style())
        btn_fullscreen.clicked.connect(self.toggle_fullscreen_text)
        layout.addWidget(btn_fullscreen)


        # story_text (بديل story_text و)
        self.story_text = QTextEdit()
        self.story_text.setReadOnly(True)
        self.story_text.setStyleSheet("background: #FFF8E1; color: black; font-size: 18px; padding: 8px;")
        self.story_text.setFixedHeight(200)
        layout.addWidget(self.story_text)

        self.btn_read = QPushButton("تم القراءة / Mark as Read")
        self.btn_read.setStyleSheet(self._gold_button_style())
        self.btn_read.clicked.connect(self.mark_as_read)
        layout.addWidget(self.btn_read, alignment=Qt.AlignCenter)

        self.lbl_footer = QLabel()
        self.lbl_footer.setAlignment(Qt.AlignCenter)
        self.lbl_footer.setStyleSheet("color: #FFD700; font-size: 12px; font-style: italic;")
        layout.addWidget(self.lbl_footer)

        self.setCentralWidget(central)
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        lang_menu = self.menuBar().addMenu("Language")
        for lang in ["English", "العربية"]:
            act = QAction(lang, self)
            act.triggered.connect(lambda checked, l=lang: self.set_language(l))
            lang_menu.addAction(act)

        self.update_texts()

    def _apply_style(self):
        self.setStyleSheet("QMainWindow { background-color: #111111; } QLabel { color: #FFD700; }")
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("#111111"))
        self.setPalette(pal)

    def _gold_button_style(self):
        return ("QPushButton { background-color: #FFD700; color: black; font-weight: bold; "
                "border-radius: 8px; padding: 6px; } QPushButton:hover { background-color: #E0C200; }")
    
    def toggle_window_size(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def update_texts(self):
        t = texts[self.language]
        self.lbl_welcome.setText(t["welcome"].format(self.user_name))
        self.btn_load.setText(t["upload"])
        self.btn_analyze.setText(t["analyze"])
        self.btn_delete.setText(t["delete"])
        self.btn_video.setText(t["video"])
        self.btn_camera.setText(t["camera"])
        self.btn_back.setText(t["back"])
        self.lbl_image.setText(t["no_image"])
        self.lbl_footer.setText(t["created"])
        self.story_text.setPlaceholderText(t["notes"])
        self.btn_read.setText(t["read_done"].format(self.user_name))
        self.btn_snapshot.setText(t["snapshot"])

    # ================= IMAGE / ANALYSIS =================
    def load_image(self):
        self.stop_video_camera()
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self.current_image_path = path
        pix = QPixmap(path)
        if pix.isNull():
            QMessageBox.warning(self, "Load failed", "Could not load image.")
            return
        self.lbl_image.setPixmap(pix.scaled(self.lbl_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.story_text.clear()
        self.log(f"Loaded image: {path}")

    def delete_image(self):
        self.stop_video_camera()
        self.current_image_path = None
        self.lbl_image.clear()
        self.lbl_image.setText(texts[self.language]["no_image"])
        self.story_text.clear()
        self.log("Image cleared.")

        

    def analyze_media(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "No file", "Please load an image or video first.")
            return

        # تحميل الموديل لو مش متحمل
        if YOLO_AVAILABLE and not self.model:
            try:
                model_path = MODEL_PATHS.get(self.model_name)
                self.model = YOLO(model_path)
                self.log(f"{self.model_name} model loaded.")
            except Exception as e:
                QMessageBox.critical(self, "Model Error", f"Failed to load {self.model_name} model:\n{e}")
                self.model = None
                return

        # نحدد إذا كان صورة أو فيديو
        ext = self.current_image_path.split(".")[-1].lower()
        if ext in ["jpg", "jpeg", "png", "bmp"]:  # صورة
            self._analyze_image_file(self.current_image_path)
        elif ext in ["mp4", "avi", "mov", "mkv"]:  # فيديو
            self._analyze_video_file(self.current_image_path)
        else:
            QMessageBox.warning(self, "Invalid file", "Unsupported file type.")

    def _analyze_image_file(self, image_path):
        img = cv2.imread(image_path)
        results = self.model(img)
        if len(results) == 0 or len(results[0].boxes) == 0:
            self.story_text.setPlainText("No landmark detected.")
            return

        img_draw = img.copy()
        detected_texts = []
        for box in results[0].boxes:
            cls_idx = int(box.cls[0])
            cls_name = self.model.names[cls_idx]
            hist = get_history(cls_name)
            detected_texts.append(f"{cls_name}: {hist}")

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img_draw, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        self._update_display(img_draw, detected_texts)


    def _analyze_video_file(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "Video Error", "Failed to open video file.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model(frame)
            img_draw = frame.copy()
            detected_texts = []

            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls_idx = int(box.cls[0])
                    cls_name = self.model.names[cls_idx]
                    hist = get_history(cls_name)
                    detected_texts.append(f"{cls_name}: {hist}")

                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(img_draw, cls_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            self._update_display(img_draw, detected_texts)
            QApplication.processEvents()

        cap.release()

    def _update_display(self, img_draw, detected_texts):
        h, w, ch = img_draw.shape
        bytes_per_line = ch * w
        qimg = QImage(img_draw.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.lbl_image.setPixmap(QPixmap.fromImage(qimg).scaled(self.lbl_image.size(), Qt.KeepAspectRatio))

        self.story_text.setPlainText("\n".join(detected_texts))

    def toggle_fullscreen_text(self):
        if not hasattr(self, "fullscreen_window") or self.fullscreen_window is None:
            self.fullscreen_window = QWidget()
            self.fullscreen_window.setWindowTitle("Full Story View")
            self.fullscreen_window.showFullScreen()  # يفتح الشاشة كلها
            layout = QVBoxLayout(self.fullscreen_window)
            # Text Area (خط كبير للقراءة)
            text_area = QTextEdit()
            text_area.setReadOnly(True)
            text_area.setStyleSheet("""
                background: #FFF8E1;
                color: black;
                font-size: 32px;   /* ← خط كبير */
                padding: 20px;
                font-weight: bold;
            """)
            text_area.setText(self.story_text.toPlainText())
            layout.addWidget(text_area)
            # زر الرجوع
            btn_close = QPushButton("🔙 Back to App")
            btn_close.setStyleSheet("""
                QPushButton {
                    background-color: #FFD700;
                    color: black;
                    font-size: 24px;
                    padding: 12px 20px;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #FFB300;
                }
            """)
            btn_close.clicked.connect(self.close_fullscreen_text)
            layout.addWidget(btn_close, alignment=Qt.AlignCenter)
        else:
            self.fullscreen_window.showFullScreen()

    def close_fullscreen_text(self):
        if self.fullscreen_window:
            self.fullscreen_window.close()
            self.fullscreen_window = None

    def process_video_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_video_camera()
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_draw = frame_rgb.copy()
        detected_texts = []

        if YOLO_AVAILABLE and self.model:
            results = self.model(frame_rgb)
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls_idx = int(box.cls[0])
                    cls_name = self.model.names[cls_idx]
                    hist = get_history(cls_name)

                    detected_texts.append(f"{cls_name}: {hist}")

                    # هنا نفس التعديل
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(img_draw, cls_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        self.story_text.setPlainText("\n".join(detected_texts))
        h, w, ch = img_draw.shape
        bytes_per_line = ch * w
        qimg = QImage(img_draw.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.lbl_image.setPixmap(QPixmap.fromImage(qimg).scaled(self.lbl_image.size(), Qt.KeepAspectRatio))


    # ================= MARK AS READ =================
    def mark_as_read(self):
        current_text = self.story_text.toPlainText()
        if not current_text.strip():
            QMessageBox.warning(self, "No data", "No landmark detected to mark as read.")
            return
        first_line = current_text.split("\n")[0]
        landmark_name = first_line.split(":")[0]
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([self.user_name, landmark_name, timestamp])
        QMessageBox.information(self, "Marked", texts[self.language]["read_done"].format(self.user_name))
        self.log(f"Landmark '{landmark_name}' marked as read.")

    # ================= VIDEO / CAMERA =================
    def load_video(self):
        self.stop_video_camera()
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Video Error", "Cannot open video file.")
            return
        self.log(f"Loaded video: {path}")
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_video_frame)
        self.timer.start(30)

    def load_story_data(self):
        global story_data
        if self.language == "العربية":
            json_path = AR_JSON_PATH
        else:
            json_path = EN_JSON_PATH

        if Path(json_path).exists():
            with open(json_path, "r", encoding="utf-8") as f:
                story_data = json.load(f)
        else:
            story_data = {}
            print(f"No story data found for {self.language}.")
    def open_camera(self):
        if self.cap:  # لو الكاميرا مفتوحة نقفلها
            self.stop_video_camera()
            self.log("Camera closed.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Camera Error", "Cannot open webcam.")
            self.cap = None
            return
        self.log("Camera opened.")
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_video_frame)
        self.timer.start(30)

    def take_snapshot(self):
        if not self.cap:
            QMessageBox.warning(self, "Snapshot Error", "Camera/Video not running.")
            return
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "Snapshot Error", "Failed to capture frame.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Snapshot", "", "Images (*.png *.jpg)")
        if filename:
            cv2.imwrite(filename, frame)
            QMessageBox.information(self, "Saved", f"Snapshot saved to:\n{filename}")
            self.log(f"Snapshot saved: {filename}")

        if YOLO_AVAILABLE and self.model:
            results = self.model(frame_rgb)
            if len(results) > 0 and len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes):
                    cls_idx = int(box.cls[0])
                    cls_name = self.model.names[cls_idx]
                    hist = get_history(cls_name)

                    detected_texts.append(f"{cls_name}: {hist}")

                    xyxy = box.xyxy[i].cpu().numpy().astype(int)
                    cv2.rectangle(img_draw, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 2)
                    cv2.putText(img_draw, cls_name, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        self.story_text.setPlainText("\n".join(detected_texts))
        h, w, ch = img_draw.shape
        bytes_per_line = ch * w
        qimg = QImage(img_draw.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.lbl_image.setPixmap(QPixmap.fromImage(qimg).scaled(self.lbl_image.size(), Qt.KeepAspectRatio))

    def stop_video_camera(self):
        if self.timer:
            self.timer.stop()
            self.timer = None
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def toggle_expand(self):
        if self.story_text.maximumHeight() == 2000:
            # رجع الحجم الطبيعي
            self.story_text.setMaximumHeight(200)
            self.btn_expand.setText("📰 Expand / عرض كامل")
        else:
            # كبر المساحة
            self.story_text.setMaximumHeight(2000)
            self.btn_expand.setText("🔙 Collapse / تصغير")

    # ================= LANGUAGE / LOG =================
    def set_language(self, lang):
        self.language = lang
        self.update_texts()
    
        # تحديث القصص حسب اللغة الجديدة
        global story_data
        if self.language == "العربية":
            JSON_PATH = AR_JSON_PATH
        else:
            JSON_PATH = EN_JSON_PATH
    
        if Path(JSON_PATH).exists():
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                story_data = json.load(f)
        else:
            story_data = []
    
        QMessageBox.information(self, "Language Changed", f"Language switched to {lang}.")

    def log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.status.showMessage(f"[{ts}] {text}", 7000)

    def go_back(self):
        self.stop_video_camera()
        self.close()
        self.welcome_window.show()

# ------------------------ RUN APP ------------------------
def run_app():
    app = QApplication(sys.argv)
    windows = {}

    def start_callback(name, lang, model_name, welcome_ref):
        global story_data
        # تحديد JSON حسب اللغة
        if lang == "العربية":
            JSON_PATH = AR_JSON_PATH
        else:
            JSON_PATH = EN_JSON_PATH

        # تحميل البيانات
        if Path(JSON_PATH).exists():
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                story_data = json.load(f)
        else:
            story_data = []

        main = MainWindow(name, lang, model_name, welcome_ref)
        windows["main"] = main
        main.show()
        welcome_ref.hide()


    welcome = WelcomeWindow(start_callback)
    windows["welcome"] = welcome
    welcome.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_app()

    