"""
Hotel Management AI Agent
=========================
A chat-based front-desk assistant that can add, view, update and delete
customer records, using Claude's tool-calling (function calling) to decide
which action to take. Data is stored permanently in MongoDB.

Run:
    py -m pip install anthropic pymongo
    set ANTHROPIC_API_KEY=sk-ant-...
    python HotelAgent.py

Maps to the project brief:
  FR1 Collect details   -> system prompt rules + validation errors returned to model
  FR2 Save               -> save_customer tool
  FR3 View                -> find_customer tool
  FR4 Update             -> update_customer tool
  FR5 Delete (confirm)   -> model asks yes/no in chat BEFORE calling delete_customer
  FR6 Validation          -> validate_* functions, called inside every tool
  FR7 Permanent storage  -> MongoDB (hotel_db.customers), survives restarts
  FR8 Honest replies      -> tools return real success/error, model must relay them
  FR9 Stay on topic       -> system prompt instructs the agent to redirect
"""

import json
import os
import re
from datetime import datetime


import anthropic
from pymongo import MongoClient, ReturnDocument

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://admin:1234@cluster0.xcmkrdn.mongodb.net/?appName=Cluster0")
DB_NAME = "hotel_db"
MODEL = "claude-sonnet-5"

# ---------------------------------------------------------------------------
# 1. DATABASE (permanent storage - FR7)
# ---------------------------------------------------------------------------

_mongo_client = None

def get_db():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client[DB_NAME]


def get_next_sequence_value(sequence_name="customer_id"):
    db = get_db()
    counter = db["counters"].find_one_and_update(
        {"_id": sequence_name},
        {"$inc": {"sequence_value": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return counter["sequence_value"]


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

    db = get_db()
    new_id = get_next_sequence_value("customer_id")
    customer_doc = {
        "id": new_id,
        "full_name": full_name.strip(),
        "phone": phone.strip(),
        "email": (email or "").strip() or None,
        "room_number": room_number.strip(),
        "checkin": checkin.strip(),
        "checkout": checkout.strip(),
        "special_requests": (special_requests or "").strip() or None,
    }
    db["customers"].insert_one(customer_doc)
    return {"success": True, "customer": doc_to_dict_by_id(new_id)}


def doc_to_dict_by_id(customer_id):
    db = get_db()
    try:
        cid = int(customer_id)
    except (ValueError, TypeError):
        return None
    doc = db["customers"].find_one({"id": cid})
    if not doc:
        return None
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


def tool_find_customer(query):
    db = get_db()
    query_str = str(query).strip()
    if query_str.isdigit():
        docs = list(db["customers"].find({"id": int(query_str)}))
    else:
        escaped_query = re.escape(query_str)
        docs = list(db["customers"].find({"full_name": {"$regex": escaped_query, "$options": "i"}}))

    matches = []
    for d in docs:
        item = dict(d)
        item.pop("_id", None)
        matches.append(item)

    if not matches:
        return {"success": True, "matches": [], "message": "No customer found matching that."}
    return {"success": True, "matches": matches}


def tool_update_customer(customer_id, updates):
    existing = doc_to_dict_by_id(customer_id)
    if not existing:
        return {"success": False, "errors": [f"No customer with ID {customer_id} exists."]}

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

    db = get_db()
    clean_updates = {k: (v.strip() if isinstance(v, str) else v) for k, v in updates.items()}
    db["customers"].update_one({"id": int(customer_id)}, {"$set": clean_updates})
    return {
        "success": True,
        "changed_fields": updates,
        "customer": doc_to_dict_by_id(customer_id),
    }


def tool_delete_customer(customer_id):
    existing = doc_to_dict_by_id(customer_id)
    if not existing:
        return {"success": False, "errors": [f"No customer with ID {customer_id} exists."]}
    db = get_db()
    db["customers"].delete_one({"id": int(customer_id)})
    return {"success": True, "deleted_customer": existing}


def tool_list_customers():
    db = get_db()
    docs = list(db["customers"].find().sort("id", 1))
    customers = []
    for d in docs:
        item = dict(d)
        item.pop("_id", None)
        customers.append(item)
    return {"success": True, "customers": customers}


TOOL_FUNCTIONS = {
    "save_customer": tool_save_customer,
    "find_customer": tool_find_customer,
    "update_customer": tool_update_customer,
    "delete_customer": tool_delete_customer,
    "list_customers": tool_list_customers,
}

# ---------------------------------------------------------------------------
# 4. TOOL SCHEMAS (given to Claude)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "save_customer",
        "description": "Save a brand new customer record once ALL required details have been collected "
                        "and look valid. Do not call this until you have full_name, phone, room_number, "
                        "checkin and checkout. Never invent values for missing fields.",
        "input_schema": {
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
    {
        "name": "find_customer",
        "description": "Look up existing customer(s) by numeric ID or by (partial) name.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "update_customer",
        "description": "Update one or more fields of an existing customer. The customer ID itself "
                        "can never be changed. Only include fields that are actually changing.",
        "input_schema": {
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
    {
        "name": "delete_customer",
        "description": "Permanently delete a customer record. IMPORTANT: only call this after you have "
                        "already asked the staff member for explicit confirmation in plain chat "
                        "('do you really want to delete...?') and they replied yes. If they reply no, "
                        "do not call this tool at all.",
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "integer"}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "list_customers",
        "description": "List every customer currently stored (bonus feature).",
        "input_schema": {"type": "object", "properties": {}},
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
- For deletion: first show which customer you are about to delete and ask a clear yes/no \
question in chat. Only call delete_customer after the staff member replies with something \
affirmative like "yes". If they say "no", cancel and say so - do not call the tool.
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
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return {"success": False, "errors": [f"Unknown tool: {name}"]}
    try:
        return func(**tool_input)
    except TypeError as e:
        return {"success": False, "errors": [f"Bad arguments for {name}: {e}"]}


def get_anthropic_api_key():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key and os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ANTHROPIC_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass
    while not api_key:
        print("ANTHROPIC_API_KEY environment variable is not set.")
        api_key = input("Please enter your Anthropic API key (sk-ant-...): ").strip()
    return api_key


def main():
    api_key = get_anthropic_api_key()
    client = anthropic.Anthropic(api_key=api_key)
    messages = []

    print("Hotel front-desk assistant. Type 'quit' to exit.\n")

    while True:
        user_text = input("Staff: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})

        # Loop until Claude stops asking to use tools
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b.text for b in response.content if b.type == "text"]

            if text_blocks:
                print("Agent:", " ".join(text_blocks))

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in tool_uses:
                result = run_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    }
                )
            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    main()