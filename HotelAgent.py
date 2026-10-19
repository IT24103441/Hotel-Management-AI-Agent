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


def load_local_env():
    """Load simple KEY=value settings from .env without overriding the shell."""
    if not os.path.exists(".env"):
        return
    try:
        with open(".env", "r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


load_local_env()
DATABASE_PATH = os.getenv("DATABASE_PATH", "hotel.db")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
_pending_delete = None

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
                checkin TEXT NOT NULL,
                checkout TEXT NOT NULL,
                special_requests TEXT
            )
        """)


def row_to_dict(row):
    return dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# 2. VALIDATION (FR6 - reject invalid data, explain why)
# ---------------------------------------------------------------------------

PHONE_RE = re.compile(r"^\+?\d{9,15}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ROOM_RE = re.compile(r"^[A-Za-z]?\d{1,4}[A-Za-z]?$")


def validate_full_name(v):
    if not v or not v.strip():
        return "Full name cannot be empty."
    return None


def validate_phone(v):
    if not v or not PHONE_RE.match(v.strip()):
        return "Phone number must be 9 to 15 digits, and may start with '+'."
    return None


def validate_email(v):
    if v is None or v.strip() == "":
        return None  # optional field
    if not EMAIL_RE.match(v.strip()):
        return "That email address doesn't look valid (expected something like name@example.com)."
    return None


def validate_room(v):
    if not v or not ROOM_RE.match(v.strip()):
        return "Room number should look like '204' or 'A12'."
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


def validate_new_customer(full_name, phone, email, room_number, checkin, checkout):
    errors = []
    for msg in (
        validate_full_name(full_name),
        validate_phone(phone),
        validate_email(email),
        validate_room(room_number),
    ):
        if msg:
            errors.append(msg)
    date_msg = validate_dates(checkin, checkout)
    if date_msg:
        errors.append(date_msg)
    return errors


# ---------------------------------------------------------------------------
# 3. TOOL IMPLEMENTATIONS
#    Each returns a plain dict: {"success": bool, ...}. The model must only
#    report success to staff if success is True (FR8 - honest replies).
# ---------------------------------------------------------------------------

def tool_save_customer(full_name, phone, room_number, checkin, checkout,
                        email=None, special_requests=None):
    errors = validate_new_customer(full_name, phone, email, room_number, checkin, checkout)
    if errors:
        return {"success": False, "errors": errors}

    with get_db() as db:
        cursor = db.execute(
            """INSERT INTO customers
               (full_name, phone, email, room_number, checkin, checkout, special_requests)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (full_name.strip(), phone.strip(), (email or "").strip() or None,
             room_number.strip(), checkin.strip(), checkout.strip(),
             (special_requests or "").strip() or None),
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

    allowed = {"full_name", "phone", "email", "room_number", "checkin", "checkout", "special_requests"}
    unknown = set(updates.keys()) - allowed
    if unknown:
        return {"success": False, "errors": [f"Unknown field(s): {', '.join(unknown)}"]}

    merged = {**existing, **updates}
    errors = validate_new_customer(
        merged["full_name"], merged["phone"], merged.get("email"),
        merged["room_number"], merged["checkin"], merged["checkout"],
    )
    if errors:
        return {"success": False, "errors": errors}

    clean_updates = {k: (v.strip() if isinstance(v, str) else v) for k, v in updates.items()}
    columns = ", ".join(f"{key} = ?" for key in clean_updates)
    values = list(clean_updates.values()) + [int(customer_id)]
    with get_db() as db:
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
    except sqlite3.Error as e:
        return {"success": False, "errors": [f"The local database could not complete the deletion: {e}"]}
    return {"success": True, "deleted_customer": existing}


def tool_list_customers():
    with get_db() as db:
        customers = [row_to_dict(row) for row in db.execute(
            "SELECT * FROM customers ORDER BY id"
        ).fetchall()]
    return {"success": True, "customers": customers}


TOOL_FUNCTIONS = {
    "save_customer": tool_save_customer,
    "find_customer": tool_find_customer,
    "update_customer": tool_update_customer,
    "delete_customer": tool_delete_customer,
    "list_customers": tool_list_customers,
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
                            "checkin and checkout. Never invent values for missing fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name": {"type": "string"},
                    "phone": {"type": "string", "description": "9 to 15 digits, may start with +"},
                    "email": {"type": "string", "description": "optional"},
                    "room_number": {"type": "string", "description": "e.g. 204 or A12"},
                    "checkin": {"type": "string", "description": "YYYY-MM-DD"},
                    "checkout": {"type": "string", "description": "YYYY-MM-DD, after checkin"},
                    "special_requests": {"type": "string", "description": "optional free text"},
                },
                "required": ["full_name", "phone", "room_number", "checkin", "checkout"],
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
                            "can never be changed. Only include fields that are actually changing.",
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
- Required fields for a new customer: full name, phone number, room number, check-in date, \
check-out date. Email and special requests are optional.
- Before calling save_customer, quietly make sure you already have every required field from \
the conversation.
- If a tool reports validation errors, explain the problem in simple, friendly words and ask \
the staff member to correct it. Do not say something was saved/changed/deleted unless the tool \
result says success: true.
- For updates, clearly say what changed afterwards. The customer ID itself can never change.
- For deletion: call delete_customer for the requested customer. The application will show \
the customer and require the staff member's clear yes/no answer before deleting. Do not claim \
the record was deleted unless the application confirms success.
- If several customers match a search, list them briefly and ask which one is meant.
- Keep replies polite, short and professional.
- If asked about anything unrelated to customer records (e.g. room pricing, billing, general \
chit-chat), politely explain that you only handle guest records (add, view, update, delete) and \
briefly say what you can help with.
"""

# ---------------------------------------------------------------------------
# 5. CHAT LOOP
# ---------------------------------------------------------------------------

def run_tool(name, tool_input):
    global _pending_delete
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
    global _pending_delete
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

    messages.append({"role": "user", "content": user_text})

    # Loop until Ollama returns a normal text response without tool calls.
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
