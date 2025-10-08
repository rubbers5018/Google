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

# --- Image Processing Engine ---
# Note: These functions will be called from a separate thread.

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
        rotation = osd['rotate']
        if rotation == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 180: return cv2.rotate(image, cv2.ROTATE_180)
        if rotation == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
    except (TesseractError, ValueError):
        (h, w) = image.shape[:2]
        if w > h: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image

def is_front_card(image, saturation_threshold=40):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_saturation = np.mean(hsv[:, :, 1])
    return mean_saturation > saturation_threshold

def enhance_card(image, clahe_clip, clahe_grid, bilateral_d, bilateral_sigma):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    final_image = cv2.bilateralFilter(final_image, bilateral_d, bilateral_sigma, bilateral_sigma)
    return final_image

def process_image(image_path, output_dir, args):
    try:
        image = cv2.imread(image_path)
        if image is None: return 0
        (h, w) = image.shape[:2]
        if h > args.max_size or w > args.max_size:
            r = args.max_size / float(max(h, w))
            dim = (int(w * r), args.max_size) if h > w else (args.max_size, int(w * r))
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        original_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, args.thresh_block_size, 2)
        kernel = np.ones((5,5),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_cards = 0
        for i, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
            if cv2.contourArea(contour) < args.min_area: continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                found_cards += 1
                warped = perspective_transform(original_image, approx.reshape(4, 2))
                oriented = auto_orient_card(warped)
                resized = cv2.resize(oriented, (args.width, args.height), interpolation=cv2.INTER_AREA)
                enhanced = enhance_card(resized, args.clahe_clip, args.clahe_grid, args.bilateral_d, args.bilateral_sigma)
                card_type = "front" if is_front_card(enhanced) else "back"
                base_filename, ext = os.path.splitext(os.path.basename(image_path))
                output_filename = f"{base_filename}_card_{found_cards}_{card_type}{ext}"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, enhanced)
        return found_cards
    except Exception as e:
        logging.error(f"Failed to process {os.path.basename(image_path)}: {e}")
        return 0

# --- GUI Classes ---

class DragDropWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drag & Drop Image Files Here")
        self.setStyleSheet("border: 2px dashed #aaa; border-radius: 5px; font-size: 16px; color: #888;")
    def dragEnterEvent(self, event): event.acceptProposedAction() if event.mimeData().hasUrls() else event.ignore()
    def dropEvent(self, event): event.acceptProposedAction()

class ProcessingWorker(QObject):
    """A QObject worker that runs the image processing in a separate thread."""
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, file_paths, output_dir, args):
        super().__init__()
        self.file_paths = file_paths
        self.output_dir = output_dir
        self.args = args

    def run(self):
        """The main processing loop."""
        for i, file_path in enumerate(self.file_paths):
            logging.info(f"Processing: {file_path}")
            process_image(file_path, self.output_dir, self.args)
            self.progress.emit(i + 1)
        self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Card Scanner GUI")
        self.setGeometry(100, 100, 600, 500)
        self.setAcceptDrops(True)
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

    def update_button_state(self):
        is_list_empty = self.file_list_widget.count() == 0
        self.process_button.setEnabled(not is_list_empty)
        self.progress_bar.setVisible(not is_list_empty)
        if is_list_empty: self.progress_bar.setValue(0)

    def dragEnterEvent(self, event): self.drag_drop_widget.dragEnterEvent(event)
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            existing_items = {self.file_list_widget.item(i).text() for i in range(self.file_list_widget.count())}
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')) and file_path not in existing_items:
                        self.file_list_widget.addItem(file_path)

    def start_processing(self):
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            logging.warning("No output directory selected. Aborting.")
            return

        file_paths = [self.file_list_widget.item(i).text() for i in range(self.file_list_widget.count())]
        self.progress_bar.setMaximum(len(file_paths))
        self.progress_bar.setValue(0)
        self.process_button.setEnabled(False)
        self.file_list_widget.setEnabled(False)
        self.drag_drop_widget.setEnabled(False)

        args = SimpleNamespace(
            width=250, height=350, min_area=5000, max_size=1024,
            thresh_block_size=15, clahe_clip=3.0, clahe_grid=10,
            bilateral_d=11, bilateral_sigma=100
        )

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

    def on_processing_finished(self):
        logging.info("Processing complete.")
        QMessageBox.information(self, "Success", "All images have been processed successfully.")
        self.process_button.setEnabled(True)
        self.file_list_widget.setEnabled(True)
        self.drag_drop_widget.setEnabled(True)
        self.progress_bar.setValue(self.progress_bar.maximum())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        pytesseract.get_tesseract_version()
    except TesseractError:
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Icon.Critical)
        error_msg.setText("Tesseract OCR Not Found")
        error_msg.setInformativeText("Tesseract is not installed or not in your system's PATH. Please install it and try again.")
        error_msg.setWindowTitle("Error")
        error_msg.exec()
        sys.exit(1)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())