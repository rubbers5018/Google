import sys
import os
import logging
import cv2
import numpy as np
import pytesseract
from pytesseract import TesseractError
from types import SimpleNamespace

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QListWidget, QPushButton, QProgressBar, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

# --- Image Processing Engine ---

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def auto_orient_card(image):
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT, config='--psm 0')
        rotation = osd.get('rotate', 0)
        if rotation == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 180: return cv2.rotate(image, cv2.ROTATE_180)
        if rotation == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
    except (TesseractError, ValueError, KeyError):
        h, w = image.shape[:2]
        if w > h: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image

def is_front_card(image, saturation_threshold=40):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_saturation = np.mean(hsv[:, :, 1])
    return mean_saturation > saturation_threshold

# --- Adaptive Enhancement Pipeline ---

def classify_material(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    mean = np.sum(hist * np.arange(256))
    variance = np.sum(hist * (np.arange(256) - mean)**2)
    std_dev = np.sqrt(variance)

    if std_dev > 70: return "chrome"
    if std_dev < 50 and 80 < mean < 180: return "cardboard"
    return "refractory"

def enhance_chrome(image, args):
    logging.info("Applying CHROME enhancement pipeline.")
    # Reduce specular highlights
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    # Standard enhancement on the result
    return enhance_cardboard(inpainted, args, gamma=1.1) # Use cardboard as a base

def enhance_cardboard(image, args, gamma=1.2):
    logging.info("Applying CARDBOARD enhancement pipeline.")
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(image, table)
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=args.clahe_clip, tileGridSize=(args.clahe_grid, args.clahe_grid))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    denoised = cv2.bilateralFilter(enhanced_bgr, args.bilateral_d, args.bilateral_sigma, args.bilateral_sigma)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened

def enhance_refractory(image, args):
    logging.info("Applying REFRACTORY enhancement pipeline.")
    # A balanced approach
    return enhance_cardboard(image, args, gamma=1.15)

def adaptive_enhance_card(image, args):
    material_type = classify_material(image)
    if material_type == "chrome":
        return enhance_chrome(image, args)
    elif material_type == "cardboard":
        return enhance_cardboard(image, args)
    else: # refractory
        return enhance_refractory(image, args)

# --- Main Processing Function ---

def process_image(image_path, output_dir, args):
    try:
        logging.info(f"Processing: {os.path.basename(image_path)}")
        image = cv2.imread(image_path)
        if image is None: return 0

        h, w = image.shape[:2]
        if max(h, w) > args.max_size:
            scale = args.max_size / max(h, w)
            dim = (int(w * scale), int(h * scale))
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        original_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, args.thresh_block_size, 2)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours: return 0
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        cards_found = 0
        for contour in contours:
            if cv2.contourArea(contour) < args.min_area: continue
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            warped = perspective_transform(original_image, box)
            if warped.shape[0] < 50 or warped.shape[1] < 50: continue

            cards_found += 1
            oriented = auto_orient_card(warped)
            resized = cv2.resize(oriented, (args.width, args.height), interpolation=cv2.INTER_AREA)
            enhanced = adaptive_enhance_card(resized, args) # Use adaptive enhancement
            card_type = "front" if is_front_card(enhanced) else "back"
            base_name, ext = os.path.splitext(os.path.basename(image_path))
            output_filename = f"{base_name}_card_{cards_found}_{card_type}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, enhanced)
            logging.info(f"Saved: {output_filename}")

        return cards_found

    except Exception as e:
        logging.error(f"Error processing {os.path.basename(image_path)}: {e}", exc_info=True)
        return 0

# --- GUI Classes ---

class DragDropWidget(QLabel):
    filesDropped = pyqtSignal(list)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drag & Drop Image Files Here\n(Supports PNG, JPG, JPEG, BMP, WEBP)")
        self.setStyleSheet("border: 2px dashed #aaa; border-radius: 8px; font-size: 16px; color: #666;")
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile() and os.path.splitext(u.toLocalFile())[1].lower() in SUPPORTED_EXTENSIONS]
            if paths: self.filesDropped.emit(paths)

class ProcessingWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    def __init__(self, file_paths, output_dir, args):
        super().__init__()
        self.file_paths = file_paths
        self.output_dir = output_dir
        self.args = args
    def run(self):
        for i, file_path in enumerate(self.file_paths, 1):
            process_image(file_path, self.output_dir, self.args)
            self.progress.emit(i)
        self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Card Scanner Pro")
        self.setGeometry(100, 100, 650, 550)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        self.drag_drop_widget = DragDropWidget()
        self.file_list_widget = QListWidget()
        self.process_button = QPushButton("Start Processing")
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.drag_drop_widget, 1)
        self.layout.addWidget(self.file_list_widget, 2)
        self.layout.addWidget(self.process_button)
        self.layout.addWidget(self.progress_bar)
        self.update_button_state()
        self.file_list_widget.model().rowsInserted.connect(self.update_button_state)
        self.file_list_widget.model().rowsRemoved.connect(self.update_button_state)
        self.process_button.clicked.connect(self.start_processing)
        self.drag_drop_widget.filesDropped.connect(self.handle_files_dropped)

    def update_button_state(self):
        enabled = self.file_list_widget.count() > 0
        self.process_button.setEnabled(enabled)
        self.progress_bar.setVisible(enabled)
        if not enabled: self.progress_bar.setValue(0)

    def handle_files_dropped(self, file_paths):
        existing = {self.file_list_widget.item(i).text() for i in range(self.file_list_widget.count())}
        for fp in file_paths:
            if fp not in existing: self.file_list_widget.addItem(fp)

    def start_processing(self):
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir: return
        os.makedirs(output_dir, exist_ok=True)
        file_paths = [self.file_list_widget.item(i).text() for i in range(self.file_list_widget.count())]
        if not file_paths: return
        self.progress_bar.setMaximum(len(file_paths))
        self.progress_bar.setValue(0)
        self.set_ui_enabled(False)
        args = SimpleNamespace(width=250, height=350, min_area=5000, max_size=1024, thresh_block_size=15, clahe_clip=3.0, clahe_grid=8, bilateral_d=11, bilateral_sigma=75)
        self.thread = QThread()
        self.worker = ProcessingWorker(file_paths, output_dir, args)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.start()

    def set_ui_enabled(self, enabled):
        self.process_button.setEnabled(enabled)
        self.file_list_widget.setEnabled(enabled)
        self.drag_drop_widget.setEnabled(enabled)

    def on_processing_finished(self):
        self.set_ui_enabled(True)
        self.update_button_state()
        QMessageBox.information(self, "Processing Complete", "All images have been processed successfully!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        pytesseract.get_tesseract_version()
    except TesseractError:
        QMessageBox.critical(None, "Tesseract Not Found", "Tesseract OCR is not installed or not in PATH.")
        sys.exit(1)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())