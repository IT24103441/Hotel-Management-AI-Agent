"""
Hotel Management AI Agent
=========================
A chat-based front-desk assistant that can add, view, update and delete
customer records, using Ollama's tool-calling (function calling) to decide
which action to take. Ollama runs locally by default; data is stored in SQLite.

Run:
    py -m pip install -r requirements.txt
    python HotelAgentWeb.py  # open the browser chat frontend
    python HotelAgent.py     # use the terminal chat instead

The default model is the already-installed llama3.2:3b. Set OLLAMA_MODEL in .env
to another model shown by `ollama list` if you prefer.

For Ollama Cloud, set OLLAMA_HOST=https://ollama.com, provide OLLAMA_API_KEY,
and set OLLAMA_MODEL to a model available in your Ollama Cloud account.

Maps to the project brief:
  FR1 Collect details   -> system prompt rules + validation errors returned to model
  FR2 Save               -> save_customer tool
  FR3 View                -> find_customer tool
  FR4 Update             -> update_customer tool
  FR5 Delete (confirm)   -> application requires a clear yes before deleting
  FR6 Validation          -> validate_* functions, called inside every tool
  FR7 Permanent storage  -> SQLite (hotel.db), survives restarts
  FR8 Honest replies      -> tools return real success/error, model must relay them
  FR9 Stay on topic       -> system prompt instructs the agent to redirect
"""

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime


from ollama import Client

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def load_local_env():
    """Load simple KEY=value settings from .env without overriding the shell."""
    env_path = os.path.join(APP_DIR, ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


load_local_env()
DATABASE_PATH = os.getenv("DATABASE_PATH", os.path.join(APP_DIR, "hotel.db"))
if not os.path.isabs(DATABASE_PATH):
    DATABASE_PATH = os.path.join(APP_DIR, DATABASE_PATH)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
_pending_delete = None
_pending_cancel = None

# ---------------------------------------------------------------------------
# 1. DATABASE (permanent local storage - FR7)
# ---------------------------------------------------------------------------

@contextmanager
def get_db():
    db = sqlite3.connect(DATABASE_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialize_database():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                email TEXT,
                room_number TEXT NOT NULL,
                guest_count INTEGER NOT NULL DEFAULT 1,
                checkin TEXT NOT NULL,
                checkout TEXT NOT NULL,
                special_requests TEXT,
                booking_status TEXT NOT NULL DEFAULT 'reserved',
                nightly_rate INTEGER NOT NULL DEFAULT 0,
                total_amount INTEGER NOT NULL DEFAULT 0
            )
        """)
        customer_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(customers)").fetchall()
        }
        migrations = {
            "guest_count": "INTEGER NOT NULL DEFAULT 1",
            "booking_status": "TEXT NOT NULL DEFAULT 'reserved'",
            "nightly_rate": "INTEGER NOT NULL DEFAULT 0",
            "total_amount": "INTEGER NOT NULL DEFAULT 0",
        }
        for column, definition in migrations.items():
            if column not in customer_columns:
                db.execute(f"ALTER TABLE customers ADD COLUMN {column} {definition}")
        db.execute("""
            CREATE TABLE IF NOT EXISTS rooms (
                room_number INTEGER PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'available'
                    CHECK (status IN ('available', 'maintenance'))
            )
        """)
        db.executemany(
            "INSERT OR IGNORE INTO rooms (room_number, status) VALUES (?, 'available')",
            ((room_number,) for room_number in range(1, 501)),
        )
        db.execute("""
            CREATE TABLE IF NOT EXISTS room_rates (
                room_type TEXT PRIMARY KEY,
                nightly_rate INTEGER NOT NULL CHECK (nightly_rate >= 0)
            )
        """)
        db.executemany(
            "INSERT OR IGNORE INTO room_rates (room_type, nightly_rate) VALUES (?, ?)",
            (("double", 10000), ("triple", 15000), ("six_guest", 25000)),
        )
        ensure_phone_unique_index(db)


def row_to_dict(row):
    return dict(row) if row is not None else None


def ensure_phone_unique_index(db):
    """Add the database constraint once legacy duplicate numbers are resolved."""
    duplicates = db.execute(
        "SELECT 1 FROM customers GROUP BY phone HAVING COUNT(*) > 1 LIMIT 1"
    ).fetchone()
    if duplicates is None:
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_customers_phone ON customers(phone)")


# ---------------------------------------------------------------------------
# 2. VALIDATION (FR6 - reject invalid data, explain why)
# ---------------------------------------------------------------------------

PHONE_RE = re.compile(r"^(?:[0-9]{10}|\+94[0-9]{9})$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_full_name(v):
    if not v or not v.strip():
        return "Full name cannot be empty."
    return None


def validate_phone(v):
    if not isinstance(v, str) or not PHONE_RE.fullmatch(v.strip()):
        return "Enter a 10-digit local number (e.g. 0771234567) or +94 followed by 9 digits (e.g. +94771234567)."
    return None


def normalize_phone(phone):
    """Store Sri Lankan +94 numbers in the equivalent 10-digit local format."""
    cleaned = phone.strip()
    return "0" + cleaned[3:] if cleaned.startswith("+94") else cleaned


def validate_email(v):
    if v is None or v.strip() == "":
        return None  # optional field
    if not EMAIL_RE.match(v.strip()):
        return "That email address doesn't look valid (expected something like name@example.com)."
    return None


def validate_room(v):
    try:
        room_number = int(str(v).strip())
    except (TypeError, ValueError):
        return "Choose a room number from 1 to 500."
    if not 1 <= room_number <= 500:
        return "Choose a room number from 1 to 500."
    return None


def room_capacity(room_number):
    number = int(str(room_number).strip())
    if 1 <= number <= 199:
        return 2
    if 200 <= number <= 299:
        return 3
    if 300 <= number <= 500:
        return 6
    return None


def room_type_for_number(room_number):
    number = int(str(room_number).strip())
    if 1 <= number <= 199:
        return "double"
    if 200 <= number <= 299:
        return "triple"
    if 300 <= number <= 500:
        return "six_guest"
    return None


def validate_guest_count(guest_count, room_number):
    if isinstance(guest_count, bool):
        return "Guest count must be a whole number."
    if isinstance(guest_count, int):
        count = guest_count
    elif isinstance(guest_count, str) and re.fullmatch(r"[0-9]+", guest_count.strip()):
        count = int(guest_count.strip())
    else:
        return "Guest count must be a whole number."
    if count < 1:
        return "At least one guest is required for a booking."
    capacity = room_capacity(room_number)
    if capacity is not None and count > capacity:
        return f"Room {room_number} can hold up to {capacity} guests. Choose a room with enough capacity."
    return None


def parse_date(v):
    try:
        return datetime.strptime(v.strip(), "%Y-%m-%d")
    except (ValueError, AttributeError):
        return None


def validate_dates(checkin, checkout):
    ci = parse_date(checkin)
    co = parse_date(checkout)
    if ci is None:
        return "Check-in date must be in YYYY-MM-DD format."
    if co is None:
        return "Check-out date must be in YYYY-MM-DD format."
    if co <= ci:
        return "Check-out date must be after the check-in date."
    return None


BOOKING_STATUSES = {"reserved", "checked_in", "checked_out", "cancelled"}


def booking_nights(checkin, checkout):
    return (parse_date(checkout) - parse_date(checkin)).days


def get_nightly_rate(db, room_number):
    room_type = room_type_for_number(room_number)
    row = db.execute(
        "SELECT nightly_rate FROM room_rates WHERE room_type = ?", (room_type,)
    ).fetchone()
    return int(row["nightly_rate"]) if row else 0


def room_is_in_maintenance(db, room_number):
    row = db.execute(
        "SELECT status FROM rooms WHERE room_number = ?", (int(room_number),)
    ).fetchone()
    return row is None or row["status"] == "maintenance"


def validate_new_customer(full_name, phone, email, room_number, guest_count, checkin, checkout):
    errors = []
    for msg in (
        validate_full_name(full_name),
        validate_phone(phone),
        validate_email(email),
        validate_room(room_number),
    ):
        if msg:
            errors.append(msg)
    if not validate_room(room_number):
        guest_msg = validate_guest_count(guest_count, room_number)
        if guest_msg:
            errors.append(guest_msg)
    date_msg = validate_dates(checkin, checkout)
    if date_msg:
        errors.append(date_msg)
    return errors


def find_room_booking_conflict(db, room_number, checkin, checkout, exclude_customer_id=None):
    """Find a stay in the same room that overlaps the requested date range."""
    sql = """SELECT id, full_name, room_number, checkin, checkout
             FROM customers
             WHERE room_number = ? COLLATE NOCASE
               AND checkin < ?
               AND checkout > ?
               AND booking_status IN ('reserved', 'checked_in')"""
    params = [room_number.strip(), checkout.strip(), checkin.strip()]
    if exclude_customer_id is not None:
        sql += " AND id != ?"
        params.append(int(exclude_customer_id))
    return row_to_dict(db.execute(sql, params).fetchone())


def find_phone_conflict(db, phone, exclude_customer_id=None):
    normalized = normalize_phone(phone)
    international = "+94" + normalized[1:]
    sql = "SELECT id, full_name FROM customers WHERE phone IN (?, ?)"
    params = [normalized, international]
    if exclude_customer_id is not None:
        sql += " AND id != ?"
        params.append(int(exclude_customer_id))
    return row_to_dict(db.execute(sql, params).fetchone())


def phone_conflict_error(conflict):
    return (
        f"That phone number is already used by customer #{conflict['id']} "
        f"({conflict['full_name']}). Each customer must have a unique phone number."
    )


def room_conflict_error(room_number, conflict):
    return (
        f"Room {room_number} is already booked for those dates by "
        f"{conflict['full_name']} (check-in {conflict['checkin']}, "
        f"check-out {conflict['checkout']}). Choose another room or different dates."
    )


# ---------------------------------------------------------------------------
# 3. TOOL IMPLEMENTATIONS
#    Each returns a plain dict: {"success": bool, ...}. The model must only
#    report success to staff if success is True (FR8 - honest replies).
# ---------------------------------------------------------------------------

def tool_save_customer(full_name, phone, room_number, guest_count, checkin, checkout,
                        email=None, special_requests=None):
    errors = validate_new_customer(full_name, phone, email, room_number, guest_count, checkin, checkout)
    if errors:
        return {"success": False, "errors": errors}

    normalized_room = str(int(str(room_number).strip()))
    with get_db() as db:
        # Lock before checking availability so concurrent requests cannot both
        # see the room as free and create overlapping stays.
        db.execute("BEGIN IMMEDIATE")
        phone_conflict = find_phone_conflict(db, phone)
        if phone_conflict:
            return {"success": False, "errors": [phone_conflict_error(phone_conflict)]}
        if room_is_in_maintenance(db, normalized_room):
            return {"success": False, "errors": [f"Room {normalized_room} is marked for maintenance. Choose another room."]}
        conflict = find_room_booking_conflict(db, normalized_room, checkin, checkout)
        if conflict:
            return {"success": False, "errors": [room_conflict_error(room_number, conflict)]}
        normalized_phone = normalize_phone(phone)
        nightly_rate = get_nightly_rate(db, normalized_room)
        total_amount = nightly_rate * booking_nights(checkin, checkout)
        cursor = db.execute(
            """INSERT INTO customers
               (full_name, phone, email, room_number, guest_count, checkin, checkout,
                special_requests, booking_status, nightly_rate, total_amount)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?)""",
            (full_name.strip(), normalized_phone, (email or "").strip() or None,
             normalized_room, int(guest_count), checkin.strip(), checkout.strip(),
             (special_requests or "").strip() or None, nightly_rate, total_amount),
        )
        customer = row_to_dict(db.execute(
            "SELECT * FROM customers WHERE id = ?", (cursor.lastrowid,)
        ).fetchone())
    return {"success": True, "customer": customer}


def doc_to_dict_by_id(customer_id):
    try:
        cid = int(customer_id)
    except (ValueError, TypeError):
        return None
    with get_db() as db:
        return row_to_dict(db.execute(
            "SELECT * FROM customers WHERE id = ?", (cid,)
        ).fetchone())


def tool_find_customer(query):
    query_str = str(query).strip()
    if not query_str:
        return {"success": False, "errors": ["Enter a customer ID or name to search for."]}
    if query_str.isdigit():
        sql, params = "SELECT * FROM customers WHERE id = ?", (int(query_str),)
    else:
        sql, params = "SELECT * FROM customers WHERE instr(lower(full_name), lower(?)) > 0 ORDER BY id", (query_str,)
    with get_db() as db:
        matches = [row_to_dict(row) for row in db.execute(sql, params).fetchall()]

    if not matches:
        return {"success": True, "matches": [], "message": "No customer found matching that."}
    return {"success": True, "matches": matches}


def tool_update_customer(customer_id, updates):
    existing = doc_to_dict_by_id(customer_id)
    if not existing:
        return {"success": False, "errors": [f"No customer with ID {customer_id} exists."]}

    if not isinstance(updates, dict) or not updates:
        return {"success": False, "errors": ["Provide at least one field to update."]}
    if "id" in updates or "customer_id" in updates or "_id" in updates:
        return {"success": False, "errors": ["The customer ID can never be changed."]}

    allowed = {"full_name", "phone", "email", "room_number", "guest_count", "checkin", "checkout", "special_requests", "booking_status"}
    unknown = set(updates.keys()) - allowed
    if unknown:
        return {"success": False, "errors": [f"Unknown field(s): {', '.join(unknown)}"]}

    merged = {**existing, **updates}
    errors = []
    if "full_name" in updates:
        error = validate_full_name(merged["full_name"])
        if error:
            errors.append(error)
    if "phone" in updates:
        error = validate_phone(merged["phone"])
        if error:
            errors.append(error)
    if "email" in updates:
        error = validate_email(merged.get("email"))
        if error:
            errors.append(error)
    if "room_number" in updates:
        error = validate_room(merged["room_number"])
        if error:
            errors.append(error)
    if "room_number" in updates or "guest_count" in updates:
        if not validate_room(merged["room_number"]):
            error = validate_guest_count(merged["guest_count"], merged["room_number"])
            if error:
                errors.append(error)
    if "checkin" in updates or "checkout" in updates:
        error = validate_dates(merged["checkin"], merged["checkout"])
        if error:
            errors.append(error)
    if "booking_status" in updates:
        if not isinstance(merged["booking_status"], str) or merged["booking_status"].strip().lower() not in BOOKING_STATUSES:
            errors.append("Booking status must be reserved, checked_in, checked_out, or cancelled.")
        else:
            merged["booking_status"] = merged["booking_status"].strip().lower()
    if errors:
        return {"success": False, "errors": errors}

    if "room_number" in updates or "guest_count" in updates:
        merged["room_number"] = str(int(str(merged["room_number"]).strip()))
    clean_updates = {k: (v.strip() if isinstance(v, str) else v) for k, v in updates.items()}
    if "phone" in clean_updates:
        clean_updates["phone"] = normalize_phone(clean_updates["phone"])
    if "room_number" in clean_updates:
        clean_updates["room_number"] = str(int(clean_updates["room_number"]))
    if "guest_count" in clean_updates:
        clean_updates["guest_count"] = int(clean_updates["guest_count"])
    if "booking_status" in clean_updates:
        clean_updates["booking_status"] = clean_updates["booking_status"].strip().lower()
    if {"room_number", "checkin", "checkout"} & set(clean_updates):
        with get_db() as db:
            rate = get_nightly_rate(db, merged["room_number"])
        nights = booking_nights(merged["checkin"], merged["checkout"])
        clean_updates["nightly_rate"] = rate
        clean_updates["total_amount"] = rate * nights
    with get_db() as db:
        db.execute("BEGIN IMMEDIATE")
        if "phone" in clean_updates:
            phone_conflict = find_phone_conflict(
                db, clean_updates["phone"], exclude_customer_id=customer_id
            )
            if phone_conflict:
                return {"success": False, "errors": [phone_conflict_error(phone_conflict)]}
        if merged.get("booking_status", "reserved") in {"reserved", "checked_in"}:
            if room_is_in_maintenance(db, merged["room_number"]):
                return {"success": False, "errors": [f"Room {merged['room_number']} is marked for maintenance."]}
            conflict = find_room_booking_conflict(
                db,
                merged["room_number"],
                merged["checkin"],
                merged["checkout"],
                exclude_customer_id=customer_id,
            )
            if conflict:
                return {"success": False, "errors": [room_conflict_error(merged['room_number'], conflict)]}
        columns = ", ".join(f"{key} = ?" for key in clean_updates)
        values = list(clean_updates.values()) + [int(customer_id)]
        db.execute(f"UPDATE customers SET {columns} WHERE id = ?", values)
    return {
        "success": True,
        "changed_fields": updates,
        "customer": doc_to_dict_by_id(customer_id),
    }


def tool_delete_customer(customer_id):
    """Delete only after the chat loop has verified an explicit yes."""
    existing = doc_to_dict_by_id(customer_id)
    if not existing:
        return {"success": False, "errors": [f"No customer with ID {customer_id} exists."]}
    try:
        with get_db() as db:
            db.execute("DELETE FROM customers WHERE id = ?", (int(customer_id),))
            ensure_phone_unique_index(db)
    except sqlite3.Error as e:
        return {"success": False, "errors": [f"The local database could not complete the deletion: {e}"]}
    return {"success": True, "deleted_customer": existing}


def tool_list_customers():
    with get_db() as db:
        customers = [row_to_dict(row) for row in db.execute(
            "SELECT * FROM customers ORDER BY id"
        ).fetchall()]
    return {"success": True, "customers": customers}


def tool_set_room_maintenance(room_number, maintenance):
    if validate_room(room_number):
        return {"success": False, "errors": ["Choose a room number from 1 to 500."]}
    if not isinstance(maintenance, bool):
        return {"success": False, "errors": ["Maintenance must be true or false."]}
    room_number = int(str(room_number).strip())
    with get_db() as db:
        db.execute("BEGIN IMMEDIATE")
        if maintenance:
            active_booking = db.execute(
                """SELECT id FROM customers
                   WHERE CAST(room_number AS INTEGER) = ?
                     AND booking_status IN ('reserved', 'checked_in')
                     AND checkout >= date('now', 'localtime')
                   LIMIT 1""",
                (room_number,),
            ).fetchone()
            if active_booking:
                return {
                    "success": False,
                    "errors": [f"Room {room_number} has an active or future booking. Cancel or complete that booking before maintenance."],
                }
        status = "maintenance" if maintenance else "available"
        db.execute("UPDATE rooms SET status = ? WHERE room_number = ?", (status, room_number))
    return {"success": True, "room_number": room_number, "status": status}


def get_room_availability(checkin, checkout):
    date_error = validate_dates(checkin, checkout)
    if date_error:
        return {"success": False, "error": date_error}

    occupied = {"double": set(), "triple": set(), "six_guest": set()}
    maintenance = {"double": set(), "triple": set(), "six_guest": set()}
    with get_db() as db:
        bookings = db.execute(
            """SELECT DISTINCT room_number FROM customers
               WHERE booking_status IN ('reserved', 'checked_in')
                 AND checkin < ? AND checkout > ?""",
            (checkout.strip(), checkin.strip()),
        ).fetchall()
        maintenance_rooms = db.execute(
            "SELECT room_number FROM rooms WHERE status = 'maintenance'"
        ).fetchall()

    for booking in bookings:
        try:
            room_number = int(str(booking["room_number"]).strip())
        except (TypeError, ValueError):
            continue
        if 1 <= room_number <= 199:
            occupied["double"].add(room_number)
        elif 200 <= room_number <= 299:
            occupied["triple"].add(room_number)
        elif 300 <= room_number <= 500:
            occupied["six_guest"].add(room_number)

    for room in maintenance_rooms:
        room_number = int(room["room_number"])
        category = room_type_for_number(room_number)
        if category:
            maintenance[category].add(room_number)

    ranges = {"double": range(1, 200), "triple": range(200, 300), "six_guest": range(300, 501)}
    totals = {category: len(room_numbers) for category, room_numbers in ranges.items()}
    return {
        "success": True,
        "checkin": checkin,
        "checkout": checkout,
        "available": {
            category: len(set(room_numbers) - occupied[category] - maintenance[category])
            for category, room_numbers in ranges.items()
        },
        "totals": totals,
        "rooms": {
            category: {
                "available": sorted(set(room_numbers) - occupied[category] - maintenance[category]),
                "booked": sorted(occupied[category]),
                "maintenance": sorted(maintenance[category]),
            }
            for category, room_numbers in ranges.items()
        },
    }


TOOL_FUNCTIONS = {
    "save_customer": tool_save_customer,
    "find_customer": tool_find_customer,
    "update_customer": tool_update_customer,
    "delete_customer": tool_delete_customer,
    "list_customers": tool_list_customers,
    "set_room_maintenance": tool_set_room_maintenance,
}

# ---------------------------------------------------------------------------
# 4. TOOL SCHEMAS (given to Ollama)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_customer",
            "description": "Save a brand new customer record once ALL required details have been collected "
                            "and look valid. Do not call this until you have full_name, phone, room_number, "
                            "guest_count, checkin and checkout. The room must be available for the full date range. "
                            "The phone must be a 10-digit local number or +94 followed by 9 digits, and be unique. Names may repeat. "
                            "Never invent values for missing fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name": {"type": "string"},
                    "phone": {"type": "string", "description": "10-digit local format (0771234567) or +94 format (+94771234567); unique across customers"},
                    "email": {"type": "string", "description": "optional"},
                    "room_number": {"type": "string", "description": "Choose a room from 1 to 500: 1-199 double (up to 2 guests), 200-299 triple (up to 3), 300-500 (up to 6)"},
                    "guest_count": {"type": "integer", "description": "Number of guests; must fit the selected room's capacity"},
                    "checkin": {"type": "string", "description": "YYYY-MM-DD"},
                    "checkout": {"type": "string", "description": "YYYY-MM-DD, after checkin"},
                    "special_requests": {"type": "string", "description": "optional free text"},
                },
                "required": ["full_name", "phone", "room_number", "guest_count", "checkin", "checkout"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_customer",
            "description": "Look up existing customer(s) by numeric ID or by (partial) name.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_customer",
            "description": "Update one or more fields of an existing customer. The customer ID itself "
                            "can never be changed. Only include fields that are actually changing. "
                            "Phone numbers must use a 10-digit local format or +94 followed by 9 digits, and be unique across customers. "
                            "If changing room_number or guest_count, the guests must fit the room capacity. "
                            "booking_status may be reserved, checked_in, checked_out, or cancelled. "
                            "The resulting room and stay dates must not overlap another booking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "updates": {
                        "type": "object",
                        "description": "Map of field name to new value, e.g. {\"room_number\": \"305\"}",
                    },
                },
                "required": ["customer_id", "updates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_customer",
            "description": "Start deleting a customer record after the staff member requests deletion. "
                            "The application will show the customer and require an explicit yes/no "
                            "confirmation before it performs the deletion.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "integer"}},
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_room_maintenance",
            "description": "Mark a room as under maintenance or return it to service. Do not put a room into maintenance if it has an active or future booking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_number": {"type": "integer", "description": "A room number from 1 to 500"},
                    "maintenance": {"type": "boolean", "description": "true to mark for maintenance; false to return it to service"},
                },
                "required": ["room_number", "maintenance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_customers",
            "description": "List every customer currently stored (bonus feature).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = """You are the front-desk assistant for a small hotel. Staff talk to you in plain \
sentences and you manage customer records using your tools.

Rules you must always follow:
- Ask for missing required details in small steps (one or two at a time), never a long form.
- Never guess or invent a value for a missing or unclear field.
- When an action is needed, call the provided tool directly. Never print tool-call JSON as a message to staff.
- Phone numbers must use either a 10-digit local format (e.g. 0771234567) or +94 followed by 9 digits (e.g. +94771234567). Treat both formats as the same number when checking uniqueness. Different customers may have the same name.
- Required fields for a new customer: full name, phone number, room number, number of guests, check-in date, \
check-out date. Email and special requests are optional.
- Hotel rooms are numbered 1 to 500: rooms 1-199 are double rooms for up to 2 guests, rooms 200-299 are triple rooms for up to 3 guests, and rooms 300-500 hold up to 6 guests. Ask for the preferred room number and number of guests before booking. Never book a room outside its guest capacity.
- A room cannot be assigned to two customers for overlapping stay dates. If the requested \
room is already booked, explain which dates conflict and ask for another room or dates.
- Before calling save_customer, quietly make sure you already have every required field from \
the conversation.
- If a tool reports validation errors, explain the problem in simple, friendly words and ask \
the staff member to correct it. Do not say something was saved/changed/deleted unless the tool \
result says success: true.
- For updates, clearly say what changed afterwards. The customer ID itself can never change.
- Booking statuses are reserved, checked_in, checked_out, and cancelled. Cancellation preserves the booking record; never describe it as deleted.
- Before cancelling through chat, ask for a clear yes/no confirmation. The application enforces this confirmation.
- Room maintenance can be set or cleared with set_room_maintenance. An occupied or future-booked room cannot be marked for maintenance.
- The hotel has editable demo rates in LKR: double, triple, and six-guest room categories. A booking stores the current nightly rate and total (nightly rate times nights); do not invent a different price. Staff can edit the demo rates in the management panel.
- Reply in the same language and script the staff member uses, including Sinhala or Sinhala written with Latin letters (Singlish).
- For deletion: call delete_customer for the requested customer. The application will show \
the customer and require the staff member's clear yes/no answer before deleting. Do not claim \
the record was deleted unless the application confirms success.
- If several customers match a search, list them briefly and ask which one is meant.
- Keep replies polite, short and professional.
- For billing questions, report the stored nightly rate and booking total for the requested guest when available. Do not claim that a payment was collected. Staff can also review and export booking history, edit demo rates, and manage room maintenance in the web panel.
- Reply in the same language and script the staff member uses, including Sinhala or Sinhala written with Latin letters (Singlish).
- If asked about anything unrelated to customer records (e.g. room pricing, billing, general \
chit-chat), politely explain that you only handle guest records (add, view, update, delete) and \
briefly say what you can help with.
"""

# ---------------------------------------------------------------------------
# 5. CHAT LOOP
# ---------------------------------------------------------------------------

def run_tool(name, tool_input):
    global _pending_delete, _pending_cancel
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return {"success": False, "errors": [f"Unknown tool: {name}"]}
    try:
        if name == "delete_customer":
            customer_id = tool_input.get("customer_id")
            customer = doc_to_dict_by_id(customer_id)
            if not customer:
                return {"success": False, "errors": [f"No customer with ID {customer_id} exists."]}
            _pending_delete = customer
            return {
                "success": False,
                "confirmation_required": True,
                "customer": customer,
                "message": "Waiting for the staff member's explicit yes or no.",
            }
        if name == "update_customer" and isinstance(tool_input.get("updates"), dict) and tool_input["updates"].get("booking_status") == "cancelled":
            customer_id = tool_input.get("customer_id")
            customer = doc_to_dict_by_id(customer_id)
            if not customer:
                return {"success": False, "errors": [f"No customer with ID {customer_id} exists."]}
            _pending_cancel = customer
            return {"success": False, "confirmation_required": True, "customer": customer, "message": "Waiting for the staff member's explicit yes or no to cancel this booking."}
        return func(**tool_input)
    except TypeError as e:
        return {"success": False, "errors": [f"Bad arguments for {name}: {e}"]}
    except sqlite3.Error as e:
        return {"success": False, "errors": [f"The local database could not complete the request: {e}"]}
    except Exception as e:
        return {"success": False, "errors": [f"I couldn't complete that database action: {e}"]}


def create_ollama_client():
    api_key = os.getenv("OLLAMA_API_KEY")
    if OLLAMA_HOST == "https://ollama.com":
        if not api_key:
            raise RuntimeError(
                "Ollama Cloud needs OLLAMA_API_KEY. For local Ollama, use "
                "OLLAMA_HOST=http://localhost:11434 and leave the key blank."
            )
        return Client(
            host=OLLAMA_HOST,
            headers={"Authorization": f"Bearer {api_key}"},
        )
    return Client(host=OLLAMA_HOST)


def process_chat_turn(client, messages, user_text):
    """Run one chat turn and return the text replies for CLI or GUI display."""
    global _pending_delete, _pending_cancel
    replies = []

    # The application, not the model, enforces the delete confirmation.
    if _pending_delete is not None:
        answer = re.sub(r"[^a-z ]", "", user_text.casefold()).strip()
        customer = _pending_delete
        if answer in {"yes", "y", "yes please", "confirm", "confirm delete", "yes delete", "yes delete it"}:
            _pending_delete = None
            result = tool_delete_customer(customer["id"])
            reply = (
                f"Deleted customer #{customer['id']} ({customer['full_name']}, room {customer['room_number']})."
                if result.get("success") else "I couldn't delete that customer: " + "; ".join(result.get("errors", []))
            )
            messages.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply},
            ])
            return [reply]
        if answer in {"no", "n", "no thanks", "cancel", "cancel delete", "do not delete"}:
            _pending_delete = None
            reply = f"Cancelled. Customer #{customer['id']} was not deleted."
            messages.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply},
            ])
            return [reply]
        return ["Please reply yes to delete this customer, or no to cancel."]

    if _pending_cancel is not None:
        answer = re.sub(r"[^a-z ]", "", user_text.casefold()).strip()
        customer = _pending_cancel
        if answer in {"yes", "y", "yes please", "confirm", "confirm cancel", "yes cancel", "yes cancel it"}:
            _pending_cancel = None
            result = tool_update_customer(customer["id"], {"booking_status": "cancelled"})
            reply = (f"Cancelled booking #{customer['id']} for {customer['full_name']}. The booking remains in history." if result.get("success") else "I couldn't cancel that booking: " + "; ".join(result.get("errors", [])))
            messages.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}])
            return [reply]
        if answer in {"no", "n", "no thanks", "cancel", "do not cancel"}:
            _pending_cancel = None
            reply = f"Okay. Booking #{customer['id']} was not cancelled."
            messages.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}])
            return [reply]
        return ["Please reply yes to cancel this booking, or no to keep it."]

    messages.append({"role": "user", "content": user_text})

    # Loop until Ollama returns a normal text response without tool calls.
    malformed_tool_retries = 0
    while True:
        response = client.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            stream=False,
        )

        assistant_message = response.message
        messages.append(assistant_message)
        tool_calls = assistant_message.tool_calls or []
        if not tool_calls and assistant_message.content:
            looks_like_tool_call = re.search(
                r'"name"\s*:\s*"(?:save_customer|find_customer|update_customer|delete_customer|list_customers)"',
                assistant_message.content,
            )
            if looks_like_tool_call:
                messages.pop()
                if malformed_tool_retries == 0:
                    malformed_tool_retries += 1
                    messages.append({
                        "role": "user",
                        "content": "Your last reply was not executed because it was tool-call JSON in plain text. Use the provided tool directly. Do not show tool-call JSON to staff.",
                    })
                    continue
                reply = (
                    "I couldn't complete that request because the local model returned a tool request "
                    "as plain text instead of calling the hotel function. Please try the request again."
                )
                messages.append({"role": "assistant", "content": reply})
                return [reply]
        delete_requested = any(call.function.name == "delete_customer" for call in tool_calls)
        if assistant_message.content and not delete_requested:
            replies.append(assistant_message.content)

        if not tool_calls:
            return replies

        for call in tool_calls:
            tool_name = call.function.name
            result = run_tool(tool_name, call.function.arguments)
            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(result, default=str),
            })

        if _pending_delete is not None:
            customer = _pending_delete
            replies.append(
                f"Are you sure you want to delete customer #{customer['id']}, "
                f"{customer['full_name']} (room {customer['room_number']})? Reply yes or no."
            )
            return replies
        if _pending_cancel is not None:
            customer = _pending_cancel
            replies.append(f"Do you want to cancel booking #{customer['id']} for {customer['full_name']} (room {customer['room_number']})? The record will stay in history. Reply yes or no.")
            return replies


def main():
    initialize_database()
    client = create_ollama_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Hotel front-desk assistant. Type 'quit' to exit.\n")
    while True:
        user_text = input("Staff: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue
        try:
            for reply in process_chat_turn(client, messages, user_text):
                print("Agent:", reply)
        except Exception as e:
            print(f"Agent: I couldn't connect to Ollama or complete that request: {e}")


if __name__ == "__main__":
    main()
