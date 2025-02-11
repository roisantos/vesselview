import os
import subprocess
import time
import webbrowser

# Define the base log directory
log_dir = os.path.abspath("./tensorboardLog")  # Get absolute path

# Ensure the log directory exists
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Print directories for debugging
print(f"Available log directories:\n{os.listdir(log_dir)}")

# Launch TensorBoard
print(f"Starting TensorBoard with logdir: {log_dir}")
tensorboard_process = subprocess.Popen(
    ["tensorboard", "--logdir", log_dir, "--port", "6006"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Give TensorBoard time to start
time.sleep(5)

# Open in browser
webbrowser.open("http://localhost:6006")

print("TensorBoard is running. Press Ctrl+C to stop.")
try:
    tensorboard_process.wait()
except KeyboardInterrupt:
    print("Shutting down TensorBoard...")
    tensorboard_process.terminate()
