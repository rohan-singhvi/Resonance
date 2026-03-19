#!/bin/bash
# Installs resonance boot persistence on the Zo server.
# Run once: scp this directory to /home/workspace/resonance-boot/ then execute.
set -e

BOOT_DIR="/home/workspace/resonance-boot"

echo "=== Installing resonance boot hook ==="

# 1. Copy files to persistent workspace location
mkdir -p "$BOOT_DIR"
cp -v "$BOOT_DIR/patch_files.py" "$BOOT_DIR/patch_files.py" 2>/dev/null || true
cp -v "$BOOT_DIR/supervisord-user.conf" "$BOOT_DIR/supervisord-user.conf" 2>/dev/null || true

# 2. Add hook to ~/.zo_secrets (if not already present)
if ! grep -q 'RESONANCE START' ~/.zo_secrets 2>/dev/null; then
    cat >> ~/.zo_secrets << 'HOOK'

### RESONANCE START ###
# Patch space server.ts and _home.tsx before every zo-space build (fast, idempotent)
python3 /home/workspace/resonance-boot/patch_files.py >> /var/log/resonance-boot.log 2>&1
# Start resonance API supervisord if not already running
_conf=/home/workspace/resonance-boot/supervisord-user.conf
if [ -f "$_conf" ] && ! pgrep -f 'supervisord.*supervisord-user' > /dev/null 2>&1; then
    supervisord -c "$_conf" >> /var/log/resonance-boot.log 2>&1 &
fi
unset _conf
### RESONANCE END ###
HOOK
    echo "Hook added to ~/.zo_secrets"
else
    echo "Hook already in ~/.zo_secrets"
fi

# 3. Verify
echo ""
echo "=== Verification ==="
echo "patch_files.py: $(ls -la $BOOT_DIR/patch_files.py 2>/dev/null || echo MISSING)"
echo "supervisord-user.conf: $(ls -la $BOOT_DIR/supervisord-user.conf 2>/dev/null || echo MISSING)"
echo "zo_secrets hook: $(grep -c 'RESONANCE START' ~/.zo_secrets) match(es)"
echo ""
echo "Done. On next zo-space start, server.ts will be patched and resonance-api will start."
