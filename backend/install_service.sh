#!/bin/bash
# Installs a systemd user service so the backend starts automatically at login.
#
# Usage:
#   ./install_service.sh            # auto-detect device
#   ./install_service.sh gpu
#   ./install_service.sh --uninstall
set -e
cd "$(dirname "$0")"

BACKEND_DIR="$(pwd)"
SERVICE_NAME="browser-assistant-phi4"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT_PATH="$UNIT_DIR/$SERVICE_NAME.service"

if [ "$1" = "--uninstall" ]; then
    systemctl --user disable --now "$SERVICE_NAME" 2>/dev/null || true
    rm -f "$UNIT_PATH"
    systemctl --user daemon-reload
    echo "Removed $SERVICE_NAME."
    exit 0
fi

DEVICE="${1:-auto}"

if [ -x "../venv/bin/python" ]; then
    PYTHON="$(cd .. && pwd)/venv/bin/python"
else
    PYTHON="$(command -v python3)"
    echo "Warning: project venv not found; using $PYTHON"
fi

mkdir -p "$UNIT_DIR"
cat > "$UNIT_PATH" <<EOF
[Unit]
Description=Browser Assistant (Phi-4) local backend
After=network.target

[Service]
Type=simple
WorkingDirectory=$BACKEND_DIR
ExecStart=$PYTHON $BACKEND_DIR/serve.py --device $DEVICE
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME"

echo "Installed and started $SERVICE_NAME (device: $DEVICE)."
echo "Status: systemctl --user status $SERVICE_NAME"
echo "Logs:   journalctl --user -u $SERVICE_NAME -f"
echo
echo "To keep it running when you are not logged in: sudo loginctl enable-linger $USER"
