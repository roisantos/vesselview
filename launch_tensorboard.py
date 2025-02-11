import os
import subprocess
import time
import webbrowser

# Define the log directory (change this to your actual logs path)
log_dir = "./tensorboardLog"

# Ensure the log directory exists
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Launch TensorBoard
print(f"Starting TensorBoard at {log_dir}...")
tensorboard_process = subprocess.Popen(
    ["tensorboard", "--logdir", os.path.abspath(log_dir), "--port", "6006"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Give TensorBoard a few seconds to start
time.sleep(5)

# Open TensorBoard in the default web browser
webbrowser.open("http://localhost:6006")

print("TensorBoard is running. Press Ctrl+C to stop.")
try:
    tensorboard_process.wait()
except KeyboardInterrupt:
    print("Shutting down TensorBoard...")
    tensorboard_process.terminate()
