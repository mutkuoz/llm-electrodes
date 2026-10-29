#!/bin/bash
# Install the boot-time GPU fix and the neurollm resume job.   sudo bash pipeline/boot/install.sh
# Undo:  sudo systemctl disable nvidia-awake neurollm-resume && sudo rm /etc/udev/rules.d/99-nvidia-awake.rules
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
here=$(dirname "$(readlink -f "$0")")
install -m 0755 "$here/nvidia-awake.sh" /usr/local/bin/nvidia-awake.sh
install -m 0755 "$here/neurollm-resume.sh" /usr/local/bin/neurollm-resume.sh
install -m 0644 "$here/99-nvidia-awake.rules" /etc/udev/rules.d/99-nvidia-awake.rules
rm -f /etc/udev/rules.d/80-nvidia-awake.rules  # an earlier name that sorted before the driver's own rule
install -m 0644 "$here/nvidia-awake.service" "$here/neurollm-resume.service" /etc/systemd/system/
# SELinux: give the copies their standard labels, so systemd may run them.
restorecon /usr/local/bin/nvidia-awake.sh /usr/local/bin/neurollm-resume.sh /etc/udev/rules.d/99-nvidia-awake.rules \
  /etc/systemd/system/nvidia-awake.service /etc/systemd/system/neurollm-resume.service 2>/dev/null || true
udevadm control --reload
systemctl daemon-reload
systemctl enable nvidia-awake.service neurollm-resume.service
echo "Installed. After the next reboot: GPU kept awake, pending jobs resume, log in runs/boot.log."
