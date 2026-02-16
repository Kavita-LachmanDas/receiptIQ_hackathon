"""
Image Preprocessing Module
Applies advanced image processing techniques to optimize receipt images for OCR.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io


class ImageProcessor:
    """Advanced image preprocessor for receipt images."""

    def __init__(self):
        self.processing_steps = []

    def _add_step_thumbnail(self, name: str, image: np.ndarray):
        """Store a small thumbnail of each step to save memory."""
        h, w = image.shape[:2]
        max_h = 400
        if h > max_h:
            scale = max_h / h
            thumb = cv2.resize(image, (int(w * scale), max_h), interpolation=cv2.INTER_AREA)
        else:
            thumb = image.copy()
        if len(thumb.shape) == 2:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        self.processing_steps.append((name, thumb))

    def load_image(self, uploaded_file) -> np.ndarray:
        """Load image from Streamlit uploaded file or file path."""
        if isinstance(uploaded_file, str):
            image = cv2.imread(uploaded_file)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            uploaded_file.seek(0)

        if image is None:
            raise ValueError("Could not load image. Please upload a valid image file.")

        self.processing_steps = []
        self._add_step_thumbnail("Original", image)
        return image

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        self._add_step_thumbnail("Grayscale", gray)
        return gray

    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced noise reduction."""
        if len(image.shape) == 2:
            denoised = cv2.fastNlMeansDenoising(image, None, h=12, templateWindowSize=7, searchWindowSize=21)
        else:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, h=12, hForColorComponents=12)
        self._add_step_thumbnail("Noise Reduction", denoised)
        return denoised

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        self._add_step_thumbnail("Contrast Enhanced", enhanced)
        return enhanced

    def apply_thresholding(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for better text extraction."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8
        )
        self._add_step_thumbnail("Adaptive Threshold", thresh)
        return thresh

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew/rotation."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        coords = np.column_stack(np.where(gray > 0))
        if len(coords) < 5:
            return image

        try:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            if abs(angle) > 15:
                return image

            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
            self._add_step_thumbnail("Deskewed", rotated)
            return rotated
        except Exception:
            return image

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image to make text crisper."""
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        self._add_step_thumbnail("Sharpened", sharpened)
        return sharpened

    def resize_for_ocr(self, image: np.ndarray, target_height: int = 2000) -> np.ndarray:
        """Resize image to optimal size for OCR."""
        h, w = image.shape[:2]
        if h < target_height:
            scale = target_height / h
            new_w = int(w * scale)
            resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
            return resized
        return image

    def process_receipt(self, uploaded_file) -> dict:
        """
        Full preprocessing pipeline for receipt images.
        Memory-optimized: only keeps essential images, frees intermediates.
        """
        import gc
        self.processing_steps = []

        # Load
        original = self.load_image(uploaded_file)

        # Resize for better OCR
        resized = self.resize_for_ocr(original)

        # Grayscale
        gray = self.convert_to_grayscale(resized)
        del resized  # free memory

        # Noise reduction
        denoised = self.reduce_noise(gray)
        del gray

        # Contrast enhancement
        enhanced = self.enhance_contrast(denoised)
        del denoised

        # Deskew
        deskewed = self.deskew(enhanced)
        del enhanced

        # Sharpen
        sharpened = self.sharpen(deskewed)
        del deskewed

        # Threshold (for OCR)
        thresholded = self.apply_thresholding(sharpened)

        gc.collect()

        return {
            "original": original,
            "thresholded": thresholded,
            "final_for_ocr": sharpened,
            "processing_steps": self.processing_steps
        }


