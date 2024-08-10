import sys
import subprocess
import locale
import threading
import os

def handle_stream(stream, prefix):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')
    for msg in stream:
        if prefix == '[!]' and ('it/s]' in msg or 's/it]' in msg) and ('%|' in msg or 'it [' in msg):
            if msg.startswith('100%'):
                print('\r' + msg, end="", file=sys.stderr),
            else:
                print('\r' + msg[:-1], end="", file=sys.stderr),
        else:
            if prefix == '[!]':
                print(prefix, msg, end="", file=sys.stderr)
            else:
                print(prefix, msg, end="")

def process_wrap(cmd_str, cwd_path, handler=None):
    process = subprocess.Popen(cmd_str, cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    if handler is None:
        handler = handle_stream

    stdout_thread = threading.Thread(target=handler, args=(process.stdout, ""))
    stderr_thread = threading.Thread(target=handler, args=(process.stderr, "[!]"))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()

assert process_wrap([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd_path=os.path.dirname(os.path.realpath(__file__))) == 0, "ERROR: Failed to install requirements.txt. Please install them manually, and restart ComfyUI."

nodep_packages = [
    "kornia>=0.6.9",
    "xformers>=0.0.20",
]

assert process_wrap([sys.executable, "-m", "pip", "install", "--no-deps", *nodep_packages], cwd_path=os.path.dirname(os.path.realpath(__file__))) == 0, "ERROR: Failed to install last set of packages. Please install them manually, and restart ComfyUI."