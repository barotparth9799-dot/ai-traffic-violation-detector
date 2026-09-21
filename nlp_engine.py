import spacy
from datetime import datetime


class TrafficNLP:
    def __init__(self):
        """
        Initializes the spaCy model.

        The model must be installed before the application starts.
        Runtime downloading is intentionally disabled because
        Streamlit Cloud does not allow modifying the environment
        while the application is running.
        """
        self.model_name = "en_core_web_sm"

        try:
            self.nlp = spacy.load(self.model_name)
            print(f"Loaded spaCy model: {self.model_name}")

        except OSError:
            print(
                f"ERROR: spaCy model '{self.model_name}' is not installed. "
                "Install it before starting the application."
            )
            self.nlp = None

    def generate_violation_message(
        self,
        owner_name,
        vehicle_number,
        violation_type,
        fine_amount,
        date_time_str=None
    ):
        """
        Generates a natural language violation notification message.
        """

        if not date_time_str:
            date_time_str = datetime.now().strftime("%Y-%m-%d at %I:%M %p")

        # Clean inputs
        owner = owner_name.strip() if owner_name else "Vehicle Owner"
        plate = vehicle_number.strip().upper() if vehicle_number else "UNKNOWN"
        violation = (
            violation_type.strip().lower()
            if violation_type
            else "violating traffic rules"
        )

        # Core template generation
        message = (
            f"Dear {owner}, your vehicle {plate} was detected violating "
            f"traffic rules due to {violation} on {date_time_str}. "
            f"A fine of ₹{fine_amount} has been generated."
        )

        return message

    def analyze_message(self, message_text):
        """
        Runs spaCy NLP pipeline on the message text.

        Extracts:
        - Tokenization
        - Lemmatization
        - POS tagging
        - Dependency parsing
        - Named Entity Recognition
        - Custom vehicle-number detection
        """

        if not self.nlp:
            # Safe fallback if the spaCy model is unavailable
            return {
                "tokens": [
                    {
                        "text": word,
                        "pos": "N/A",
                        "explain": "Model not loaded"
                    }
                    for word in message_text.split()
                ],
                "entities": []
            }

        doc = self.nlp(message_text)

        # Token information
        tokens_info = []

        for token in doc:
            # Skip punctuation for clean POS table
            if token.is_punct:
                continue

            tokens_info.append(
                {
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "explain": spacy.explain(token.pos_)
                    or "No description",
                    "tag": token.tag_,
                    "dep": token.dep_
                }
            )

        # Named Entity Recognition
        entities_info = []

        for ent in doc.ents:
            entities_info.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "explain": spacy.explain(ent.label_)
                    or "Custom Entity"
                }
            )

        # Custom vehicle-number extraction
        # Standard spaCy model does not specifically recognize
        # Indian vehicle registration numbers.
        words = message_text.split()

        for word in words:
            word_clean = word.strip(",.")

            # Approximate vehicle number format:
            # minimum 8 characters + at least one letter + one digit
            if (
                len(word_clean) >= 8
                and any(c.isdigit() for c in word_clean)
                and any(c.isalpha() for c in word_clean)
            ):
                already_detected = any(
                    word_clean in ent["text"]
                    for ent in entities_info
                )

                if not already_detected:
                    entities_info.append(
                        {
                            "text": word_clean,
                            "label": "VEHICLE_NO",
                            "explain": (
                                "Unique vehicle registration "
                                "plate number"
                            )
                        }
                    )

        return {
            "tokens": tokens_info,
            "entities": entities_info
        }


if __name__ == "__main__":
    nlp_engine = TrafficNLP()

    msg = nlp_engine.generate_violation_message(
        "Rahul Sharma",
        "MH12AB1234",
        "riding without a helmet",
        1000
    )

    print("Generated Message:")
    print(msg)

    analysis = nlp_engine.analyze_message(msg)

    print("\n--- NLP Analysis ---")

    print("Entities Detected:")

    for ent in analysis["entities"]:
        print(
            f"  Entity: {ent['text']} -> "
            f"{ent['label']} ({ent['explain']})"
        )

    print("\nPOS Tokens Sample (first 5):")

    for token in analysis["tokens"][:5]:
        print(
            f"  Token: {token['text']} -> "
            f"POS: {token['pos']} ({token['explain']})"
        )