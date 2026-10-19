"""Local browser frontend for the hotel records agent."""

from flask import Flask, jsonify, render_template, request

import HotelAgent as agent


app = Flask(__name__)
agent.initialize_database()
ollama_client = agent.create_ollama_client()
conversation = [{"role": "system", "content": agent.SYSTEM_PROMPT}]


@app.get("/")
def home():
    return render_template("index.html", model=agent.MODEL)


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
    return jsonify({"ok": True})


@app.get("/api/availability")
def availability():
    checkin = request.args.get("checkin", "").strip()
    checkout = request.args.get("checkout", "").strip()
    result = agent.get_room_availability(checkin, checkout)
    if not result["success"]:
        return jsonify(result), 400
    return jsonify(result)


if __name__ == "__main__":
    print("Hotel chat frontend: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
