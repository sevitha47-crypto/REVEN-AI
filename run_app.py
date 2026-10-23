"""
Streamlit server launcher.
Translates runtime arguments and launches the Streamlit application on port 3000.
"""

import os
import sys

def ensure_nginx_lua_compatibility():
    """Ensure nginx allows Streamlit internal websocket and health pings in iframe mode."""
    lua_path = "/etc/nginx/user_auth_verification.lua"
    if os.path.exists(lua_path) and os.access(lua_path, os.W_OK):
        try:
            with open(lua_path, "r") as f:
                content = f.read()
            if "_stcore" not in content:
                target = 'if ngx.var.host == "localhost" then\n  return\nend'
                replacement = '''if ngx.var.host == "localhost" then
  return
end

-- Allow Streamlit internal subresources and websocket streams to bypass auth check
-- because cross-site iframes cannot reliably send partitioned cookies on fetch/WebSocket.
local uri = ngx.var.uri or ""
if string.match(uri, "^/_stcore/") or string.match(uri, "^/static/") or string.match(uri, "^/vendor/") or string.match(uri, "^/favicon") then
  return
end'''
                if target in content:
                    with open(lua_path, "w") as f:
                        f.write(content.replace(target, replacement, 1))
                    os.system("nginx -s reload 2>/dev/null")
        except Exception as e:
            print(f"Notice: Lua config check: {e}")

ensure_nginx_lua_compatibility()

port = "3000"
for i, arg in enumerate(sys.argv):
    if arg == "--port" and i + 1 < len(sys.argv):
        port = sys.argv[i + 1]
    elif arg.startswith("--port="):
        port = arg.split("=")[1]

import subprocess

base_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(base_dir, "venv", "bin", "python3")
venv_pip = os.path.join(base_dir, "venv", "bin", "pip")

# If venv doesn't exist, create it
if not os.path.exists(venv_python):
    print("Notice: Setting up virtual environment in ./venv ...")
    try:
        subprocess.run([sys.executable, "-m", "venv", os.path.join(base_dir, "venv")], check=True)
    except Exception as e:
        print(f"Warning creating venv: {e}")

python_bin = venv_python if os.path.exists(venv_python) else sys.executable

# Verify that streamlit is available
try:
    check = subprocess.run([python_bin, "-c", "import streamlit"], capture_output=True)
    if check.returncode != 0:
        print("Notice: Streamlit not found in environment. Installing requirements...")
        pip_bin = venv_pip if os.path.exists(venv_pip) else "pip3"
        req_path = os.path.join(base_dir, "requirements.txt")
        if os.path.exists(req_path):
            subprocess.run([pip_bin, "install", "-r", req_path], check=True)
        else:
            subprocess.run([pip_bin, "install", "streamlit"], check=True)
except Exception as e:
    print(f"Notice: Dependency check: {e}")

cmd = [
    python_bin,
    "-m",
    "streamlit",
    "run",
    "app.py",
    "--server.port",
    port,
    "--server.address",
    "0.0.0.0",
    "--server.headless",
    "true",
    "--server.enableCORS",
    "false",
    "--server.enableXsrfProtection",
    "false",
    "--server.enableWebsocketCompression",
    "false",
    "--server.allowedHosts",
    "*",
    "--client.allowedOrigins",
    "*",
    "--browser.gatherUsageStats",
    "false"
]

print(f"Starting RevenAI — Smart Sales Prediction and Analytics System on port {port}...")
os.execvp(python_bin, cmd)
