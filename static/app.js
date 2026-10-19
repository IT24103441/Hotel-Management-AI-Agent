const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const typing = document.querySelector("#typing");

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
