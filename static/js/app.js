// /static/js/app.js

import { AudioPlayer } from "./audio-player.js";
import { AudioRecorder } from "./audio-recorder.js";
import {
  addMessage,
  getAudioMode,
  getSessionId,
  setAgentState,
  setAudioMode,
  setMicState,
} from "./ui.js";

// Global WebSocket connection
let socket;

// Audio player and recorder
const audioPlayer = new AudioPlayer();
const audioRecorder = new AudioRecorder(sendAudio);

// Connect to the WebSocket server
function connectWebSocket(isAudio) {
  // REFACTOR 1: If a socket already exists, close it before creating a new one.
  // This ensures a clean reconnect when toggling audio mode.
  if (socket) {
    socket.close();
  }

  const sessionId = getSessionId();
  const wsUrl = `ws://${window.location.host}/ws/${sessionId}?is_audio=${isAudio}`;
  console.log(`Connecting to ${wsUrl}`);
  socket = new WebSocket(wsUrl);

  socket.onopen = (event) => {
    console.log("WebSocket connection opened.", event);
    setAgentState("idle");
    setMicState(getAudioMode() ? "idle" : "disabled");
    const startMessage = isAudio ? "(Audio mode)" : "(Text mode)";
    addMessage(startMessage, "info");
  };

  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log("Message from server ", message);

    if (message.turn_complete) {
      setAgentState("idle");
      return;
    }

    if (message.mime_type?.startsWith("audio")) {
      setAgentState("speaking");
      audioPlayer.play(message.data);
    }

    if (message.mime_type?.startsWith("text")) {
      addMessage(message.data, message.role);
    }
  };

  socket.onclose = (event) => {
    console.log("WebSocket connection closed.", event);
    setAgentState("disconnected");
    setMicState("disabled");
    socket = null; // Clear the socket variable
  };

  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
    setAgentState("disconnected");
    setMicState("disabled");
  };
}

// Send a message to the agent
function sendMessage(message) {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message));
  } else {
    console.error("WebSocket is not open. readyState: ", socket?.readyState);
    addMessage("Connection is not open. Please refresh the page.", "error");
  }
}

// Send text message to the agent
function sendTextMessage() {
  const textInput = document.getElementById("text-input");
  const text = textInput.value;
  if (!text) return;
  addMessage(text, "user");
  sendMessage({
    mime_type: "text/plain",
    data: text,
    role: "user",
  });
  textInput.value = "";
}

// Send audio message to the agent
function sendAudio(audioData) {
  sendMessage({
    mime_type: "audio/pcm",
    data: audioData,
    role: "user",
  });
}

// Toggle between audio and text mode
function toggleMic() {
  const isAudio = !getAudioMode();
  setAudioMode(isAudio);

  // REFACTOR 2: Call connectWebSocket directly instead of re-running start().
  // This will gracefully close the old connection and open a new one
  // with the correct audio mode.
  connectWebSocket(isAudio);
}

// Toggle the microphone recording state
function toggleMicState() {
  if (audioRecorder.isRecording) {
    audioRecorder.stop();
  } else {
    audioRecorder.start();
  }
}

// Main function to start the application
function start() {
  console.log("Starting ADK Voice Agent");

  const isAudio = getAudioMode();
  setAgentState("connecting");
  connectWebSocket(isAudio);

  const textInput = document.getElementById("text-input");
  const sendButton = document.getElementById("send-button");
  const micButton = document.getElementById("mic-button");
  const micState = document.getElementById("mic-state");

  sendButton.addEventListener("click", sendTextMessage);
  textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendTextMessage();
  });
  micButton.addEventListener("click", toggleMic);
  micState.addEventListener("click", toggleMicState);
}

// Start the application when the DOM is loaded
document.addEventListener("DOMContentLoaded", start);