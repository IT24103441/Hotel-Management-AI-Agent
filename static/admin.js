const themeToggle = document.querySelector("#theme-toggle");
const bookingRows = document.querySelector("#booking-rows");

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
}

function setTheme(theme, persist = true) {
  const dark = theme === "dark";
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  const icon = document.createElement("span");
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = dark ? "☀" : "☾";
  themeToggle.replaceChildren(icon, document.createTextNode(dark ? " Light mode" : " Dark mode"));
  themeToggle.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  if (persist) {
    try { localStorage.setItem("hotel-desk-theme", dark ? "dark" : "light"); } catch { /* Keep the selection for this page. */ }
  }
}

let savedTheme = "light";
try { savedTheme = localStorage.getItem("hotel-desk-theme") === "dark" ? "dark" : "light"; } catch { /* Use light mode by default. */ }
setTheme(savedTheme, false);
themeToggle.addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

async function refreshDashboard() {
  try {
    const response = await fetch("/api/dashboard");
    const data = await response.json();
    if (!response.ok) return;
    document.querySelector("#dash-available").textContent = data.available_today;
    document.querySelector("#dash-active").textContent = data.active_bookings;
    document.querySelector("#dash-checkins").textContent = data.checkins_today;
    document.querySelector("#dash-checkouts").textContent = data.checkouts_today;
  } catch { /* Dashboard values refresh when the local server responds. */ }
}

function bookingDate(value) {
  return value ? new Date(`${value}T00:00:00`).toLocaleDateString() : "—";
}

function money(value) {
  return `LKR ${Number(value || 0).toLocaleString()}`;
}

function currentBookingQuery() {
  const query = new URLSearchParams();
  const status = document.querySelector("#history-status").value;
  const from = document.querySelector("#history-from").value;
  const to = document.querySelector("#history-to").value;
  if (status) query.set("status", status);
  if (from) query.set("from", from);
  if (to) query.set("to", to);
  return query;
}

async function loadBookings() {
  const query = currentBookingQuery();
  document.querySelector("#csv-export").href = `/api/bookings.csv${query.size ? `?${query}` : ""}`;
  bookingRows.innerHTML = '<tr><td colspan="7">Loading bookings…</td></tr>';
  try {
    const response = await fetch(`/api/bookings?${query}`);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not load bookings.");
    bookingRows.replaceChildren();
    if (!result.bookings.length) {
      bookingRows.innerHTML = '<tr><td colspan="7">No bookings match these filters.</td></tr>';
      return;
    }
    for (const booking of result.bookings) {
      const row = document.createElement("tr");
      const guest = document.createElement("td");
      guest.innerHTML = `<strong>#${booking.id} · ${escapeHtml(booking.full_name)}</strong><small>${escapeHtml(booking.phone || "")}</small>`;
      const room = document.createElement("td");
      room.textContent = `Room ${booking.room_number} · ${booking.guest_count} guest${booking.guest_count === 1 ? "" : "s"}`;
      const stay = document.createElement("td");
      stay.textContent = `${bookingDate(booking.checkin)} – ${bookingDate(booking.checkout)}`;
      const statusCell = document.createElement("td");
      const select = document.createElement("select");
      select.className = "status-select";
      for (const [value, label] of [["reserved", "Reserved"], ["checked_in", "Checked in"], ["checked_out", "Checked out"], ["cancelled", "Cancelled"]]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        option.selected = (booking.booking_status || "reserved") === value;
        select.append(option);
      }
      statusCell.append(select);
      const rate = document.createElement("td"); rate.textContent = `${money(booking.nightly_rate)} / night`;
      const total = document.createElement("td"); total.textContent = money(booking.total_amount);
      const saveCell = document.createElement("td");
      const save = document.createElement("button"); save.type = "button"; save.className = "table-save"; save.textContent = "Save";
      save.addEventListener("click", async () => {
        if (select.value === "cancelled" && booking.booking_status !== "cancelled" && !window.confirm(`Cancel booking #${booking.id} for ${booking.full_name}? The record will be kept in history.`)) return;
        save.disabled = true;
        try {
          const response = await fetch(`/api/bookings/${booking.id}/status`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: select.value, confirmed: select.value === "cancelled" }) });
          const update = await response.json();
          if (!response.ok) throw new Error(update.error || "Could not update status.");
          await Promise.all([loadBookings(), refreshDashboard()]);
        } catch (error) { window.alert(error.message); save.disabled = false; }
      });
      saveCell.append(save);
      row.append(guest, room, stay, statusCell, rate, total, saveCell);
      bookingRows.append(row);
    }
  } catch (error) {
    bookingRows.innerHTML = `<tr><td colspan="7">${escapeHtml(error.message)}</td></tr>`;
  }
}

async function loadRates() {
  try {
    const response = await fetch("/api/rates");
    const data = await response.json();
    for (const category of ["double", "triple", "six_guest"]) document.querySelector(`#rate-${category}`).value = data.rates[category] ?? 0;
  } catch { document.querySelector("#rates-feedback").textContent = "Could not load rates."; }
}

async function inspectMaintenanceRoom() {
  const roomNumber = Number(document.querySelector("#maintenance-room").value);
  const button = document.querySelector("#maintenance-toggle");
  if (!Number.isInteger(roomNumber) || roomNumber < 1 || roomNumber > 500) { button.disabled = true; return; }
  try {
    const response = await fetch(`/api/rooms/${roomNumber}`);
    const room = await response.json();
    if (!response.ok) throw new Error(room.error);
    button.dataset.maintenance = String(room.status !== "maintenance");
    button.textContent = room.status === "maintenance" ? "Return to service" : "Mark for maintenance";
    button.disabled = false;
  } catch { button.disabled = true; }
}

document.querySelector("#history-refresh").addEventListener("click", loadBookings);
for (const id of ["history-status", "history-from", "history-to"]) document.querySelector(`#${id}`).addEventListener("change", loadBookings);
document.querySelector("#rates-save").addEventListener("click", async () => {
  const feedback = document.querySelector("#rates-feedback");
  const rates = Object.fromEntries(["double", "triple", "six_guest"].map((key) => [key, document.querySelector(`#rate-${key}`).value]));
  try {
    const response = await fetch("/api/rates", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ rates }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    feedback.textContent = "Demo rates saved.";
  } catch (error) { feedback.textContent = error.message || "Could not save rates."; }
});
document.querySelector("#maintenance-room").addEventListener("input", inspectMaintenanceRoom);
document.querySelector("#maintenance-toggle").addEventListener("click", async () => {
  const button = document.querySelector("#maintenance-toggle");
  const roomNumber = Number(document.querySelector("#maintenance-room").value);
  const maintenance = button.dataset.maintenance === "true";
  const feedback = document.querySelector("#maintenance-feedback");
  button.disabled = true;
  try {
    const response = await fetch(`/api/rooms/${roomNumber}/maintenance`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ maintenance }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    feedback.textContent = maintenance ? `Room ${roomNumber} marked for maintenance.` : `Room ${roomNumber} returned to service.`;
    await Promise.all([inspectMaintenanceRoom(), refreshDashboard()]);
  } catch (error) { feedback.textContent = error.message || "Could not update room."; await inspectMaintenanceRoom(); }
});

loadBookings();
loadRates();
refreshDashboard();
