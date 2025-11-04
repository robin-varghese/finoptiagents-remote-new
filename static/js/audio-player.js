/**
 * Audio Player Worklet
 */

export async function startAudioPlayerWorklet() {
  // 1. Create an AudioContext
  const audioContext = new AudioContext({
    sampleRate: 24000,
  });

  // FIX: Check if audioWorklet is supported before trying to use it.
  if (!audioContext.audioWorklet) {
    console.error("AudioWorklet is not supported in this browser or context. HTTPS or localhost is required.");
    // Return null or handle the error gracefully so the app doesn't crash.
    return [null, null];
  }

  // 2. Load your custom processor code
  const workletURL = new URL("./pcm-player-processor.js", import.meta.url);
  await audioContext.audioWorklet.addModule(workletURL);

  // 3. Create an AudioWorkletNode
  const audioPlayerNode = new AudioWorkletNode(
    audioContext,
    "pcm-player-processor"
  );

  // 4. Connect to the destination
  audioPlayerNode.connect(audioContext.destination);

  // The audioPlayerNode.port is how we send messages (audio data) to the processor
  return [audioPlayerNode, audioContext];
}