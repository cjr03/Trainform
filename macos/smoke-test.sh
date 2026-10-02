#!/usr/bin/env bash
# Runs the built app's server headlessly: bundled Python + Postgres start, the API and UI answer, shutdown is clean.
set -euo pipefail
cd "$(dirname "$0")/.."
RES=${1:-dist/dmg-root/Trainform.app}/Contents/Resources
DATA=$(mktemp -d /tmp/tf.XXXX)  # short: macOS caps the Postgres socket path at 103 bytes
URL=http://127.0.0.1:47999

(cd "$RES/backend" && PYTHONDONTWRITEBYTECODE=1 exec ../python/bin/python3 -m app.desktop --port 47999 \
  --pg-bin ../postgres/bin --static-dir ../frontend --data-dir "$DATA") &
PID=$!
trap 'kill $PID 2>/dev/null || true; rm -rf "$DATA"' EXIT

for _ in $(seq 120); do curl -fs "$URL/api/health" >/dev/null && break; sleep 1; done
curl -fs "$URL/api/health" >/dev/null
test "$(curl -s -o /dev/null -w '%{http_code}' -H 'Host: evil.example' "$URL/api/health")" = 400  # DNS rebinding
curl -fs "$URL/train" | grep -q '<div id="root">'
curl -fs -X POST "$URL/api/auth/signup" -H 'content-type: application/json' \
  -d '{"email":"smoke@example.com","username":"smoke","password":"smoke-test-pass","password_confirm":"smoke-test-pass"}' \
  | grep -q '"token"'

kill $PID
wait $PID || true
test ! -e "$DATA/postgres/postmaster.pid"
echo "smoke test passed"
