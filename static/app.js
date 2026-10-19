const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const typing = document.querySelector("#typing");
const availabilityCheckin = document.querySelector("#availability-checkin");
const availabilityCheckout = document.querySelector("#availability-checkout");
const availabilityList = document.querySelector("#availability-list");
const availabilityNote = document.querySelector("#availability-note");
const themeToggle = document.querySelector("#theme-toggle");
let expandedAvailabilityCategory = null;

function setTheme(theme, persist = true) {
  const darkMode = theme === "dark";
  document.documentElement.dataset.theme = darkMode ? "dark" : "light";
  const icon = document.createElement("span");
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = darkMode ? "☀" : "☾";
  themeToggle.replaceChildren(icon, document.createTextNode(darkMode ? " Light mode" : " Dark mode"));
  themeToggle.setAttribute("aria-label", darkMode ? "Switch to light mode" : "Switch to dark mode");
  if (persist) {
    try {
      localStorage.setItem("hotel-desk-theme", darkMode ? "dark" : "light");
    } catch {
      // Keep the current selection for this page if browser storage is unavailable.
    }
  }
}

let savedTheme = "light";
try {
  savedTheme = localStorage.getItem("hotel-desk-theme") === "dark" ? "dark" : "light";
} catch {
  // The page can still use the toggle without browser storage.
}
setTheme(savedTheme, false);
themeToggle.addEventListener("click", () => {
  setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});

function localDateString(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function setDefaultAvailabilityDates() {
  const today = new Date();
  const tomorrow = new Date(today);
  tomorrow.setDate(tomorrow.getDate() + 1);
  availabilityCheckin.value = localDateString(today);
  availabilityCheckout.value = localDateString(tomorrow);
}

async function refreshAvailability() {
  const checkin = availabilityCheckin.value;
  const checkout = availabilityCheckout.value;
  if (!checkin || !checkout) return;

  availabilityNote.textContent = "";
  availabilityList.replaceChildren();
  const loading = document.createElement("div");
  loading.className = "availability-status";
  loading.textContent = "Checking availability…";
  availabilityList.append(loading);

  try {
    const query = new URLSearchParams({ checkin, checkout });
    const response = await fetch(`/api/availability?${query}`);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Couldn't check these dates.");

    const roomTypes = [
      ["Double · 1–199", "double"],
      ["Triple · 200–299", "triple"],
      ["Up to 6 guests · 300–500", "six_guest"],
    ];
    availabilityList.replaceChildren();
    for (const [label, key] of roomTypes) {
      const group = document.createElement("section");
      group.className = "availability-group";
      const row = document.createElement("button");
      row.type = "button";
      row.className = "availability-row";
      row.setAttribute("aria-expanded", String(expandedAvailabilityCategory === key));
      row.setAttribute("aria-label", `${label}, ${result.available[key]} free. Click to see room numbers.`);
      const name = document.createElement("span");
      name.className = "availability-room-name";
      name.textContent = label;
      const count = document.createElement("span");
      count.className = "availability-room-count";
      count.textContent = `${result.available[key]} free`;
      count.title = `${result.available[key]} rooms available out of ${result.totals[key]}`;
      row.append(name, count);
      row.addEventListener("click", () => {
        expandedAvailabilityCategory = expandedAvailabilityCategory === key ? null : key;
        refreshAvailability();
      });

      const details = document.createElement("div");
      details.className = "availability-details";
      details.hidden = expandedAvailabilityCategory !== key;
      const roomSets = result.rooms[key];
      for (const [heading, roomNumbers] of [
        [`Available (${roomSets.available.length})`, roomSets.available],
        [`Booked (${roomSets.booked.length})`, roomSets.booked],
      ]) {
        const section = document.createElement("div");
        section.className = "room-number-section";
        const sectionHeading = document.createElement("div");
        sectionHeading.className = "room-number-heading";
        sectionHeading.textContent = heading;
        const numbers = document.createElement("div");
        numbers.className = "room-number-grid";
        if (roomNumbers.length) {
          for (const roomNumber of roomNumbers) {
            const chip = document.createElement("span");
            chip.className = "room-number-chip";
            chip.textContent = String(roomNumber);
            numbers.append(chip);
          }
        } else {
          numbers.textContent = "None";
          numbers.classList.add("room-number-empty");
        }
        section.append(sectionHeading, numbers);
        details.append(section);
      }

      group.append(row, details);
      availabilityList.append(group);
    }
  } catch (error) {
    availabilityList.replaceChildren();
    availabilityNote.textContent = error.message || "Couldn't check availability.";
  }
}

function addMessage(role, content) {
  const item = document.createElement("article");
  item.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "Y" : "H";

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = content;

  item.append(avatar, body);
  messages.append(item);
  item.scrollIntoView({ behavior: "smooth", block: "end" });
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

async function sendMessage(text) {
  const message = text.trim();
  if (!message || sendButton.disabled) return;

  welcome.classList.add("hidden");
  addMessage("user", message);
  input.value = "";
  resizeInput();
  sendButton.disabled = true;
  typing.classList.remove("hidden");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The request failed.");
    for (const reply of result.replies || []) addMessage("assistant", reply);
    if (result.booking_dates) {
      availabilityCheckin.value = result.booking_dates.checkin;
      availabilityCheckout.value = result.booking_dates.checkout;
    }
    refreshAvailability();
  } catch (error) {
    addMessage("assistant", error.message || "Something went wrong. Please try again.");
  } finally {
    typing.classList.add("hidden");
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    input.focus();
    resizeInput();
  });
});

document.querySelector("#new-chat").addEventListener("click", async () => {
  if (sendButton.disabled) return;
  try {
    await fetch("/api/reset", { method: "POST" });
    messages.replaceChildren();
    welcome.classList.remove("hidden");
    input.value = "";
    resizeInput();
    input.focus();
  } catch {
    addMessage("assistant", "Couldn't start a new conversation. Please try again.");
  }
});

availabilityCheckin.addEventListener("change", refreshAvailability);
availabilityCheckout.addEventListener("change", refreshAvailability);
setDefaultAvailabilityDates();
refreshAvailability();
