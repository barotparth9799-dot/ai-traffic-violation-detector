import cv2
import easyocr
import re
import numpy as np

class LicensePlateRecognizer:
    def __init__(self):
        """Initializes the EasyOCR reader. Keeps it on CPU by default but allows GPU if available."""
        # Use English for number plates
        self.reader = easyocr.Reader(['en'], gpu=False) 
        # Standard Indian Number Plate regex patterns:
        # e.g., MH12AB1234 or DL3CA5678 or KA03M9876
        self.plate_pattern = re.compile(r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}')

    def preprocess_image(self, cropped_image):
        """
        Applies basic preprocessing to improve OCR accuracy.
        1. Converts to grayscale.
        2. Resizes up to ensure characters are big enough.
        3. Applies bilateral filter to remove noise while keeping edges sharp.
        """
        if cropped_image is None or cropped_image.size == 0:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to double width/height (improves OCR on small license plate crops)
        resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        # Apply Bilateral Filter (reduces noise, preserves edges)
        filtered = cv2.bilateralFilter(resized, 11, 17, 17)
        
        return filtered

    def clean_text(self, text):
        """Cleans extracted text by converting to uppercase, stripping spaces, and removing non-alphanumeric chars."""
        if not text:
            return ""
        # Convert to uppercase
        clean = text.upper()
        # Remove spaces and non-alphanumeric characters
        clean = re.sub(r'[^A-Z0-9]', '', clean)
        return clean

    def extract_license_plate(self, image_roi):
        """
        Runs EasyOCR on a cropped region of a motorcycle or vehicle.
        Extracts, cleans, and validates text using standard format.
        """
        if image_roi is None or image_roi.size == 0:
            return None, 0.0

        # Preprocess the cropped license plate region
        processed = self.preprocess_image(image_roi)
        if processed is None:
            return None, 0.0

        try:
            # Run EasyOCR
            results = self.reader.readtext(processed)
        except Exception as e:
            print("OCR Error:", e)
            return None, 0.0

        best_plate = None
        highest_confidence = 0.0

        for bbox, text, prob in results:
            cleaned = self.clean_text(text)
            print(f"OCR Detected Raw: '{text}' -> Cleaned: '{cleaned}' (conf: {prob:.2f})")
            
            # Check if this cleaned text matches our pattern
            # Or if it's close to standard length of Indian plates (9-10 chars)
            if self.plate_pattern.match(cleaned):
                if prob > highest_confidence:
                    best_plate = cleaned
                    highest_confidence = prob
            # Fallback check: if it has standard pattern characters but OCR missed a letter/digit
            # e.g., length is 9 or 10 and starts with letters, has numbers in the middle/end
            elif 8 <= len(cleaned) <= 10:
                # We relax requirements slightly if it's the only option
                if prob > highest_confidence and prob > 0.40:
                    best_plate = cleaned
                    highest_confidence = prob

        # If we couldn't find a direct match, let's return the highest confidence text overall
        if not best_plate and results:
            # Sort by confidence
            results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
            cleaned_fallback = self.clean_text(results_sorted[0][1])
            if len(cleaned_fallback) >= 5: # reasonable length
                best_plate = cleaned_fallback
                highest_confidence = results_sorted[0][2]

        return best_plate, float(highest_confidence)

    def fuzzy_match_plate(self, detected_plate, registered_plates):
        """
        Compares the detected plate with a list of registered plates.
        Corrects minor OCR misreadings (e.g., O read as 0, I read as 1, etc.)
        using character-by-character similarity.
        """
        if not detected_plate:
            return None, 0.0
            
        detected_clean = self.clean_text(detected_plate)
        best_match = None
        max_similarity = 0.0
        
        for reg_plate in registered_plates:
            reg_clean = self.clean_text(reg_plate)
            
            # Simple Levenshtein-like character overlap distance
            matches = sum(1 for a, b in zip(detected_clean, reg_clean) if a == b)
            similarity = matches / max(len(detected_clean), len(reg_clean))
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = reg_plate
                
        # If similarity is above 75%, we consider it a match and auto-correct it!
        if max_similarity >= 0.75:
            return best_match, max_similarity
            
        return detected_clean, 1.0 # return original cleaned if no close match

if __name__ == "__main__":
    # Test stub
    recognizer = LicensePlateRecognizer()
    print("OCR Module Initialized successfully.")
