# Hotel Management AI Agent

A hotel front-desk assistant that manages guest bookings through a chat interface. It uses Ollama for AI, Flask for the web app, and SQLite to store guest records.

## Features

- Register, search, update, and delete guest records using chat.
- Check room availability for selected dates.
- View available and booked room numbers by clicking a room category in the sidebar.
- Prevent duplicate phone numbers while allowing guests to have the same name.
- Switch between dark mode and light mode.

## Room Types

| Room numbers | Type | Maximum guests |
| --- | --- | ---: |
| 1–199 | Double | 2 |
| 200–299 | Triple | 3 |
| 300–500 | Up to 6 guests | 6 |

A room cannot be booked by two guests for overlapping dates.

## Requirements

- Python 3.10 or newer
- Ollama installed and running
- An Ollama model that supports tool calling

The default model is `llama3.2:3b`. You can select another model already installed on your computer.

## Setup and Run

Install the project dependencies:

```powershell
py -m pip install -r requirements.txt
py HotelAgentWeb.py

Register a Guest
1. Select the check-in and check-out dates in the sidebar.
2. Click a room category to view available and booked room numbers.
3. Send the guest’s name, phone number, selected room number, number of guests, and stay dates in the chat.
Example:
Register a new guest: Guest Name, phone 0771234567, room 101, 2 guests, check-in 2026-10-20, check-out 2026-10-22.
