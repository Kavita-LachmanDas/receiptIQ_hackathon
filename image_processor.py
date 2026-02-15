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

        self.processing_steps = [("Original", image.copy())]
        return image

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        self.processing_steps.append(("Grayscale", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
        return gray

    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced noise reduction."""
        if len(image.shape) == 2:
            denoised = cv2.fastNlMeansDenoising(image, None, h=12, templateWindowSize=7, searchWindowSize=21)
        else:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, h=12, hForColorComponents=12)
        self.processing_steps.append(("Noise Reduction",
                                       cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR) if len(denoised.shape) == 2 else denoised))
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
        self.processing_steps.append(("Contrast Enhanced",
                                       cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) if len(enhanced.shape) == 2 else enhanced))
        return enhanced

    def apply_thresholding(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for better text extraction."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8
        )
        self.processing_steps.append(("Adaptive Threshold", cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))
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
            self.processing_steps.append(("Deskewed",
                                           cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR) if len(rotated.shape) == 2 else rotated))
            return rotated
        except Exception:
            return image

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image to make text crisper."""
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        self.processing_steps.append(("Sharpened",
                                       cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR) if len(sharpened.shape) == 2 else sharpened))
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
        Returns dict with processed images at different stages.
        """
        self.processing_steps = []

        # Load
        original = self.load_image(uploaded_file)

        # Resize for better OCR
        resized = self.resize_for_ocr(original)

        # Grayscale
        gray = self.convert_to_grayscale(resized)

        # Noise reduction
        denoised = self.reduce_noise(gray)

        # Contrast enhancement
        enhanced = self.enhance_contrast(denoised)

        # Deskew
        deskewed = self.deskew(enhanced)

        # Sharpen
        sharpened = self.sharpen(deskewed)

        # Threshold (for OCR)
        thresholded = self.apply_thresholding(sharpened)

        return {
            "original": original,
            "grayscale": gray,
            "denoised": denoised,
            "enhanced": enhanced,
            "sharpened": sharpened,
            "thresholded": thresholded,
            "final_for_ocr": sharpened,
            "processing_steps": self.processing_steps
        }


