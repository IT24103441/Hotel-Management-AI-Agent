"""Local browser frontend for the hotel records agent."""

import csv
import io
from datetime import date, timedelta

from flask import Flask, jsonify, make_response, render_template, request

import HotelAgent as agent


app = Flask(__name__)
agent.initialize_database()
ollama_client = agent.create_ollama_client()
conversation = [{"role": "system", "content": agent.SYSTEM_PROMPT}]


@app.get("/")
def home():
    return render_template("index.html", model=agent.MODEL)


@app.get("/admin")
def admin():
    return render_template("admin.html", model=agent.MODEL)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Type a message first."}), 400

    try:
        with agent.get_db() as db:
            previous_customer_id = db.execute(
                "SELECT COALESCE(MAX(id), 0) FROM customers"
            ).fetchone()[0]
        replies = agent.process_chat_turn(ollama_client, conversation, message)
        with agent.get_db() as db:
            new_booking = db.execute(
                "SELECT checkin, checkout FROM customers WHERE id > ? ORDER BY id DESC LIMIT 1",
                (previous_customer_id,),
            ).fetchone()
        result = {"replies": replies}
        if new_booking:
            result["booking_dates"] = {
                "checkin": new_booking["checkin"],
                "checkout": new_booking["checkout"],
            }
        return jsonify(result)
    except Exception as error:
        app.logger.exception("Hotel agent request failed")
        return jsonify({
            "error": f"Couldn't reach Ollama or complete that request. Check that model "
                     f"'{agent.MODEL}' is available. ({error})"
        }), 502


@app.post("/api/reset")
def reset():
    conversation[:] = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
    agent._pending_delete = None
    agent._pending_cancel = None
    return jsonify({"ok": True})


@app.get("/api/dashboard")
def dashboard():
    today = date.today()
    tomorrow = today + timedelta(days=1)
    availability = agent.get_room_availability(today.isoformat(), tomorrow.isoformat())
    with agent.get_db() as db:
        checkins = db.execute("SELECT COUNT(*) FROM customers WHERE checkin = ? AND booking_status = 'reserved'", (today.isoformat(),)).fetchone()[0]
        checkouts = db.execute("SELECT COUNT(*) FROM customers WHERE checkout = ? AND booking_status IN ('reserved', 'checked_in')", (today.isoformat(),)).fetchone()[0]
        active = db.execute("SELECT COUNT(*) FROM customers WHERE booking_status IN ('reserved', 'checked_in')").fetchone()[0]
        maintenance = db.execute("SELECT COUNT(*) FROM rooms WHERE status = 'maintenance'").fetchone()[0]
    return jsonify({"total_rooms": 500, "available_today": sum(availability.get("available", {}).values()), "checkins_today": checkins, "checkouts_today": checkouts, "active_bookings": active, "maintenance_rooms": maintenance})


def booking_rows():
    status = request.args.get("status", "").strip()
    if status and status not in agent.BOOKING_STATUSES:
        return None, "Choose a valid booking status."
    start, end = request.args.get("from", "").strip(), request.args.get("to", "").strip()
    clauses, params = [], []
    if status:
        clauses.append("booking_status = ?")
        params.append(status)
    if start:
        clauses.append("checkout > ?")
        params.append(start)
    if end:
        clauses.append("checkin < ?")
        params.append(end)
    sql = "SELECT * FROM customers" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY checkin DESC, id DESC"
    with agent.get_db() as db:
        return [dict(row) for row in db.execute(sql, params).fetchall()], None


@app.get("/api/bookings")
def bookings():
    rows, error = booking_rows()
    if error:
        return jsonify({"error": error}), 400
    return jsonify({"bookings": rows})


@app.get("/api/bookings.csv")
def bookings_csv():
    rows, error = booking_rows()
    if error:
        return jsonify({"error": error}), 400
    output = io.StringIO()
    fields = ["id", "full_name", "phone", "email", "room_number", "guest_count", "checkin", "checkout", "booking_status", "nightly_rate", "total_amount", "special_requests"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    response = make_response("\ufeff" + output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=hotel-bookings.csv"
    return response


@app.post("/api/bookings/<int:booking_id>/status")
def update_booking_status(booking_id):
    data = request.get_json(silent=True) or {}
    status = data.get("status")
    if status not in agent.BOOKING_STATUSES:
        return jsonify({"error": "Choose a valid booking status."}), 400
    if status == "cancelled" and data.get("confirmed") is not True:
        return jsonify({"error": "Confirm cancellation in the booking panel first."}), 400
    result = agent.tool_update_customer(booking_id, {"booking_status": status})
    if not result.get("success"):
        return jsonify({"error": "; ".join(result.get("errors", ["Could not update booking."]))}), 400
    return jsonify({"booking": result["customer"]})


@app.get("/api/rates")
def rates():
    with agent.get_db() as db:
        current = {row["room_type"]: row["nightly_rate"] for row in db.execute("SELECT room_type, nightly_rate FROM room_rates")}
    return jsonify({"rates": current})


@app.post("/api/rates")
def update_rates():
    data = request.get_json(silent=True) or {}
    values = data.get("rates")
    categories = {"double", "triple", "six_guest"}
    if not isinstance(values, dict) or set(values) != categories:
        return jsonify({"error": "Provide rates for double, triple, and six_guest rooms."}), 400
    try:
        clean = {key: int(value) for key, value in values.items()}
    except (TypeError, ValueError):
        return jsonify({"error": "Rates must be whole amounts in LKR."}), 400
    if any(value < 0 or value > 100000000 for value in clean.values()):
        return jsonify({"error": "Rates must be between LKR 0 and 100,000,000."}), 400
    with agent.get_db() as db:
        db.executemany("UPDATE room_rates SET nightly_rate = ? WHERE room_type = ?", ((value, key) for key, value in clean.items()))
    return jsonify({"rates": clean})


@app.post("/api/rooms/<int:room_number>/maintenance")
def room_maintenance(room_number):
    data = request.get_json(silent=True) or {}
    result = agent.tool_set_room_maintenance(room_number, data.get("maintenance"))
    if not result.get("success"):
        return jsonify({"error": "; ".join(result.get("errors", ["Could not update room status."]))}), 400
    return jsonify(result)


@app.get("/api/rooms/<int:room_number>")
def room_details(room_number):
    if not 1 <= room_number <= 500:
        return jsonify({"error": "Room number must be from 1 to 500."}), 400
    with agent.get_db() as db:
        room = db.execute("SELECT room_number, status FROM rooms WHERE room_number = ?", (room_number,)).fetchone()
    return jsonify(dict(room))


@app.get("/api/availability")
def availability():
    checkin = request.args.get("checkin", "").strip()
    checkout = request.args.get("checkout", "").strip()
    try:
        result = agent.get_room_availability(checkin, checkout)
    except Exception:
        app.logger.exception("Room availability lookup failed")
        return jsonify({"success": False, "error": "The hotel server could not load room availability. Check the server terminal and try again."}), 500
    if not result["success"]:
        return jsonify(result), 400
    return jsonify(result)


if __name__ == "__main__":
    print("Hotel chat frontend: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
