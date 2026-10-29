#!/bin/bash
# Keep every NVIDIA display GPU powered. With runtime power management the driver can deadlock while waking a
# powered-down card (stuck processes in nvidia_uvm, only a reboot clears it). Run as root at boot, once the driver has
# bound, because binding can reset the setting to "auto".
set -u
for dev in /sys/bus/pci/devices/*; do
  [ "$(cat "$dev/vendor" 2>/dev/null)" = "0x10de" ] || continue
  case "$(cat "$dev/class" 2>/dev/null)" in 0x03*) ;; *) continue ;; esac
  for _ in $(seq 1 60); do  # wait up to a minute for the driver to bind
    [ -e "$dev/driver" ] && break
    sleep 1
  done
  echo on > "$dev/power/control"
  echo "nvidia-awake: $(basename "$dev") power/control=$(cat "$dev/power/control") runtime_status=$(cat "$dev/power/runtime_status")"
done
