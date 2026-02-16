"""
OCR Engine Module
Extracts text from preprocessed receipt images using EasyOCR.
"""

import os
import numpy as np
import cv2
from typing import List, Dict

os.environ["PYTHONIOENCODING"] = "utf-8"


class OCREngine:
    """Optical Character Recognition engine for receipt text extraction."""

    def __init__(self, languages: List[str] = None):
        """Initialize OCR reader."""
        if languages is None:
            languages = ['en']
        import easyocr
        self.reader = easyocr.Reader(languages, gpu=False, verbose=False)

    def extract_text(self, image: np.ndarray) -> List[dict]:
        """
        Extract text from image with bounding box information.
        Returns list of dicts with text, confidence, and position.
        """
        results = self.reader.readtext(image)

        extracted = []
        for (bbox, text, confidence) in results:
            extracted.append({
                "text": text.strip(),
                "confidence": round(confidence, 4),
                "bbox": bbox,
                "y_position": int(np.mean([point[1] for point in bbox])),
                "x_position": int(np.mean([point[0] for point in bbox]))
            })

        # Sort by vertical position (top to bottom)
        extracted.sort(key=lambda x: (x["y_position"], x["x_position"]))
        return extracted

    def extract_raw_text(self, image: np.ndarray) -> str:
        """Extract and return raw text as a single string."""
        results = self.extract_text(image)
        lines = self._group_into_lines(results)
        return "\n".join(lines)

    def _group_into_lines(self, results: List[dict], y_threshold: int = None) -> List[str]:
        """Group text blocks into lines based on y-position proximity."""
        if not results:
            return []

        # Auto-calculate y_threshold based on text heights if not provided
        if y_threshold is None:
            heights = []
            for r in results:
                bbox = r["bbox"]
                h = abs(bbox[2][1] - bbox[0][1])  # height of bounding box
                if h > 0:
                    heights.append(h)
            if heights:
                avg_height = np.mean(heights)
                y_threshold = max(15, int(avg_height * 0.6))
            else:
                y_threshold = 20

        lines = []
        current_line = [results[0]]

        for i in range(1, len(results)):
            # Compare with the average y_position of the current line
            avg_y = np.mean([item["y_position"] for item in current_line])
            if abs(results[i]["y_position"] - avg_y) < y_threshold:
                current_line.append(results[i])
            else:
                current_line.sort(key=lambda x: x["x_position"])
                line_text = "  ".join([item["text"] for item in current_line])
                lines.append(line_text)
                current_line = [results[i]]

        if current_line:
            current_line.sort(key=lambda x: x["x_position"])
            line_text = "  ".join([item["text"] for item in current_line])
            lines.append(line_text)

        return lines

    def get_confidence_stats(self, results: List[dict]) -> dict:
        """Calculate confidence statistics for OCR results."""
        if not results:
            return {"mean": 0, "min": 0, "max": 0, "low_confidence_count": 0}

        confidences = [r["confidence"] for r in results]
        return {
            "mean": round(np.mean(confidences), 4),
            "min": round(min(confidences), 4),
            "max": round(max(confidences), 4),
            "low_confidence_count": sum(1 for c in confidences if c < 0.5),
            "total_blocks": len(results)
        }

    def visualize_ocr_results(self, image: np.ndarray, results: List[dict]) -> np.ndarray:
        """Draw bounding boxes on image to visualize OCR detections."""
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

        for result in results:
            bbox = result["bbox"]
            confidence = result["confidence"]

            if confidence > 0.8:
                color = (0, 255, 0)
            elif confidence > 0.5:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(vis_image, [pts], True, color, 2)

        return vis_image

    def process_image(self, processed_images: dict) -> dict:
        """
        Full OCR pipeline on preprocessed images.
        Tries multiple preprocessing variants and picks best result.
        """
        final_image = processed_images["final_for_ocr"]
        results_final = self.extract_text(final_image)

        thresh_image = processed_images["thresholded"]
        results_thresh = self.extract_text(thresh_image)

        stats_final = self.get_confidence_stats(results_final)
        stats_thresh = self.get_confidence_stats(results_thresh)

        if (len(results_final) >= len(results_thresh) and
                stats_final["mean"] >= stats_thresh.get("mean", 0)):
            best_results = results_final
            best_stats = stats_final
            used_image = final_image
        else:
            best_results = results_thresh
            best_stats = stats_thresh
            used_image = thresh_image

        lines = self._group_into_lines(best_results)
        raw_text = "\n".join(lines)

        vis_image = self.visualize_ocr_results(
            processed_images["original"], best_results
        )

        return {
            "raw_text": raw_text,
            "lines": lines,
            "detailed_results": best_results,
            "confidence_stats": best_stats,
            "visualization": vis_image,
            "ocr_image_used": used_image
        }
