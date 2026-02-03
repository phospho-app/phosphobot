#!/bin/sh
set -e

# Ensure log directory exists
LOG_DIR="${TWILL_ENTRYPOINT_LOG_DIR:-/root/entrypoint-logs}"
mkdir -p "$LOG_DIR"

# ─────────────────────────────────────────────────────────────
# 1. PYTHON DEPENDENCIES (phosphobot backend)
# ─────────────────────────────────────────────────────────────
echo "Installing Python dependencies for phosphobot..."
cd /root/workspace/phosphobot && uv sync --python 3.10

# ─────────────────────────────────────────────────────────────
# 2. DASHBOARD DEPENDENCIES AND BUILD
# ─────────────────────────────────────────────────────────────
echo "Installing npm dependencies for dashboard..."
cd /root/workspace/dashboard && npm install

echo "Building dashboard frontend..."
cd /root/workspace/dashboard && npm run build

# Copy built dashboard to phosphobot resources
echo "Copying dashboard dist to phosphobot resources..."
mkdir -p /root/workspace/phosphobot/resources/dist/
cp -r /root/workspace/dashboard/dist/* /root/workspace/phosphobot/resources/dist/

# ─────────────────────────────────────────────────────────────
# 3. BACKEND SERVER
# ─────────────────────────────────────────────────────────────
# Check if server is already running on port 8020
if curl -s --connect-timeout 2 http://localhost:8020/ >/dev/null 2>&1; then
  echo "Backend server already running on port 8020, skipping..."
else
  echo "Starting phosphobot backend server..."
  cd /root/workspace/phosphobot
  nohup uv run --python 3.10 python -m phosphobot.main run \
    --simulation=headless \
    --no-crash-telemetry \
    --no-usage-telemetry \
    --port=8020 \
    --host=0.0.0.0 \
    > "$LOG_DIR/phosphobot.log" 2>&1 &
  echo $! > "$LOG_DIR/phosphobot.pid"
  echo "Phosphobot backend started (PID: $(cat "$LOG_DIR/phosphobot.pid"))"
  
  # Wait for the server to be ready (up to 60 seconds)
  echo "Waiting for backend server to start..."
  for i in $(seq 1 60); do
    if curl -s --connect-timeout 2 http://localhost:8020/ >/dev/null 2>&1; then
      echo "Backend server is ready!"
      break
    fi
    sleep 1
  done
fi

echo "Environment setup complete!"
echo "- Backend API: http://0.0.0.0:8020"
echo "- Dashboard: http://0.0.0.0:8020/dashboard"
