import os
import subprocess
import time
import webbrowser

# Define the base log directory
log_dir = "./tensorboardLog"

# Ensure the log directory exists
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Detect subdirectories (each subdir is treated as a separate run)
run_dirs = [f"{name}:{os.path.join(log_dir, name)}" for name in os.listdir(log_dir)
            if os.path.isdir(os.path.join(log_dir, name))]

# Create logdir_spec argument for TensorBoard
logdir_spec = ",".join(run_dirs) if run_dirs else log_dir  # Default to base dir if empty

# Launch TensorBoard
print(f"Starting TensorBoard with logdir: {logdir_spec}")
tensorboard_process = subprocess.Popen(
    ["tensorboard", "--logdir_spec", logdir_spec, "--port", "6006"],
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
