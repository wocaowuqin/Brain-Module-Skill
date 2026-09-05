#!/usr/bin/env bash
set -euo pipefail

cleanup() {
    sudo pkill -x iperf3 2>/dev/null || true
}
trap cleanup EXIT

# Mininet cleanup kills processes named ryu-manager, so restart Ryu afterwards.
sudo mn -c >/dev/null 2>&1 || true
sudo systemctl restart ryu-controller.service

for _ in $(seq 1 50); do
    if curl -fsS http://127.0.0.1:8080/stats/switches >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
curl -fsS http://127.0.0.1:8080/stats/switches >/dev/null

commands=$(cat <<'EOF'
pingall
sh curl -fsS http://127.0.0.1:8080/stats/switches
h2 iperf3 -s -D -1
h1 iperf3 -c 10.0.0.2 -t 3
exit
EOF
)

printf '%s\n' "$commands" | sudo mn \
    --topo single,2 \
    --mac \
    --switch ovsk,protocols=OpenFlow13 \
    --controller remote,ip=127.0.0.1,port=6653
