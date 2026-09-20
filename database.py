import sqlite3
import os
from datetime import datetime

# Define database file path
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "traffic_system.db")

def get_connection():
    """Establishes and returns a connection to the SQLite database."""
    # Ensure the directory exists
    os.makedirs(DB_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes the database, creating tables and populating mock registry if empty."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Create vehicles table (Registered owners database)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vehicles (
        vehicle_number TEXT PRIMARY KEY,
        owner_name TEXT NOT NULL,
        phone TEXT NOT NULL,
        address TEXT NOT NULL
    )
    """)
    
    # 2. Create violations table (Recorded traffic offences)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_number TEXT NOT NULL,
        owner_name TEXT NOT NULL,
        phone TEXT,
        address TEXT,
        violation_type TEXT NOT NULL,
        fine_amount INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        nlp_message TEXT NOT NULL,
        evidence_image_path TEXT NOT NULL,
        challan_pdf_path TEXT NOT NULL
    )
    """)
    
    # Check if vehicles table is empty, and seed it with dummy records
    cursor.execute("SELECT COUNT(*) FROM vehicles")
    if cursor.fetchone()[0] == 0:
        mock_vehicles = [
            ("MH12AB1234", "Rahul Sharma", "+91 98765 43210", "Flat 402, Sunshine Apartments, Pune, Maharashtra - 411001"),
            ("DL3CA5678", "Priya Patel", "+91 87654 32109", "House No. 12, Sector 15, Dwarka, New Delhi - 110075"),
            ("KA03M9876", "Amit Kumar", "+91 76543 21098", "No. 45, 2nd Cross, Indiranagar, Bengaluru, Karnataka - 560038"),
            ("TN07X4321", "S. Balaji", "+91 91234 56789", "Plot 88, Anna Nagar East, Chennai, Tamil Nadu - 600102"),
            ("UP16TZ9999", "Vikram Singh", "+91 99988 77665", "Sector 62, Noida, Uttar Pradesh - 201301")
        ]
        cursor.executemany("INSERT INTO vehicles VALUES (?, ?, ?, ?)", mock_vehicles)
        conn.commit()
        print("Database seeded with default registered vehicles.")
        
    conn.close()

def get_vehicle_owner(vehicle_number):
    """
    Looks up a vehicle number in the registry.
    Cleans the input number plate by making it uppercase and stripping whitespace.
    """
    if not vehicle_number:
        return None
        
    cleaned_num = "".join(vehicle_number.split()).upper()
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT owner_name, phone, address FROM vehicles WHERE vehicle_number = ?", (cleaned_num,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "vehicle_number": cleaned_num,
            "owner_name": row[0],
            "phone": row[1],
            "address": row[2]
        }
    return None

def add_vehicle_owner(vehicle_number, owner_name, phone, address):
    """Adds or updates a vehicle owner in the registry database."""
    cleaned_num = "".join(vehicle_number.split()).upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO vehicles (vehicle_number, owner_name, phone, address)
    VALUES (?, ?, ?, ?)
    """, (cleaned_num, owner_name, phone, address))
    conn.commit()
    conn.close()
    return True

def log_violation(vehicle_number, owner_name, phone, address, violation_type, fine_amount, nlp_message, evidence_image_path, challan_pdf_path):
    """Logs a recorded traffic violation to the database."""
    cleaned_num = "".join(vehicle_number.split()).upper() if vehicle_number else "UNKNOWN"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO violations (vehicle_number, owner_name, phone, address, violation_type, fine_amount, timestamp, nlp_message, evidence_image_path, challan_pdf_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (cleaned_num, owner_name, phone, address, violation_type, fine_amount, timestamp, nlp_message, evidence_image_path, challan_pdf_path))
    
    violation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return violation_id

def get_all_violations():
    """Retrieves all logged violations from the database, sorted by latest first."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM violations ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    violations_list = []
    for r in rows:
        violations_list.append(dict(r))
    return violations_list

def get_all_vehicles():
    """Retrieves all registered vehicles in the database."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vehicles ORDER BY vehicle_number ASC")
    rows = cursor.fetchall()
    conn.close()
    
    vehicles_list = []
    for r in rows:
        vehicles_list.append(dict(r))
    return vehicles_list

if __name__ == "__main__":
    init_db()
    print("Database test run complete. Path:", DB_PATH)
    owner = get_vehicle_owner("MH12AB 1234")
    print("Test vehicle lookup result:", owner)
