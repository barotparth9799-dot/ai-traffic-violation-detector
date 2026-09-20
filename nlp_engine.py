import spacy
from datetime import datetime

class TrafficNLP:
    def __init__(self):
        """Initializes the spaCy model. Automatically downloads the model if it's not found."""
        self.model_name = "en_core_web_sm"
        try:
            self.nlp = spacy.load(self.model_name)
            print(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            print(f"spaCy model '{self.model_name}' not found. Downloading...")
            try:
                from spacy.cli import download
                download(self.model_name)
                self.nlp = spacy.load(self.model_name)
                print("Model downloaded and loaded successfully.")
            except Exception as e:
                print(f"Failed to download spaCy model: {e}")
                self.nlp = None

    def generate_violation_message(self, owner_name, vehicle_number, violation_type, fine_amount, date_time_str=None):
        """
        Generates a natural language violation notification message using details.
        
        Args:
            owner_name (str): Name of the vehicle owner.
            vehicle_number (str): Registered vehicle number.
            violation_type (str): Type of infraction (e.g. 'riding without a helmet').
            fine_amount (int): Penalty amount in INR.
            date_time_str (str): Date and time of incident. Defaults to current time if None.
        """
        if not date_time_str:
            date_time_str = datetime.now().strftime("%Y-%m-%d at %I:%M %p")
            
        # Clean inputs
        owner = owner_name.strip() if owner_name else "Vehicle Owner"
        plate = vehicle_number.strip().upper() if vehicle_number else "UNKNOWN"
        violation = violation_type.strip().lower() if violation_type else "violating traffic rules"
        
        # Core Template Generation
        message = (
            f"Dear {owner}, your vehicle {plate} was detected violating traffic rules "
            f"due to {violation} on {date_time_str}. A fine of ₹{fine_amount} has been generated."
        )
        return message

    def analyze_message(self, message_text):
        """
        Runs spaCy NLP pipeline on the message text.
        Extracts parts of speech (POS) and named entities (NER) for academic presentation.
        """
        if not self.nlp:
            # Fallback return if spaCy failed to load
            return {
                "tokens": [{"text": word, "pos": "N/A", "explain": "Model not loaded"} for word in message_text.split()],
                "entities": []
            }
            
        doc = self.nlp(message_text)
        
        # Extract Token information (Tokenization & POS tagging)
        tokens_info = []
        for token in doc:
            # Skip punctuation for clean POS table
            if token.is_punct:
                continue
            tokens_info.append({
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "explain": spacy.explain(token.pos_) or "No description",
                "tag": token.tag_,
                "dep": token.dep_
            })
            
        # Extract Named Entities (NER)
        entities_info = []
        for ent in doc.ents:
            entities_info.append({
                "text": ent.text,
                "label": ent.label_,
                "explain": spacy.explain(ent.label_) or "Custom Entity"
            })
            
        # Custom Entity Rule-based extraction (for license plates, since standard model doesn't know Indian plates)
        # We search if the license plate is in the text and flag it as a VEHICLE_NO entity if spaCy missed it
        words = message_text.split()
        for word in words:
            word_clean = word.strip(",.")
            # Check if it fits the vehicle number format (approx length and alphanumeric mix)
            if len(word_clean) >= 8 and any(c.isdigit() for c in word_clean) and any(c.isalpha() for c in word_clean):
                # Check if it is already in entities list to avoid duplication
                already_detected = any(word_clean in ent["text"] for ent in entities_info)
                if not already_detected:
                    entities_info.append({
                        "text": word_clean,
                        "label": "VEHICLE_NO",
                        "explain": "Unique vehicle registration plate number"
                    })
                    
        return {
            "tokens": tokens_info,
            "entities": entities_info
        }

if __name__ == "__main__":
    nlp_engine = TrafficNLP()
    msg = nlp_engine.generate_violation_message("Rahul Sharma", "MH12AB1234", "riding without a helmet", 1000)
    print("Generated Message:")
    print(msg)
    
    analysis = nlp_engine.analyze_message(msg)
    print("\n--- NLP Analysis ---")
    print("Entities Detected:")
    for ent in analysis["entities"]:
        print(f"  Entity: {ent['text']} -> {ent['label']} ({ent['explain']})")
    
    print("\nPOS Tokens Sample (first 5):")
    for token in analysis["tokens"][:5]:
        print(f"  Token: {token['text']} -> POS: {token['pos']} ({token['explain']})")
