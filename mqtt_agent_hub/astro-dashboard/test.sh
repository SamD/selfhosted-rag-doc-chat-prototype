#!/usr/bin/env bash
set -euo pipefail
# Smoke test: build and verify key daisyUI classes appear in output.

echo "=== MQTT dashboard smoke test ==="

npx astro build --silent 2>&1

HTML="dist/index.html"
failures=0

check() {
  local file="${3:-$HTML}"
  if grep -q "$2" "$file" 2>/dev/null; then
    echo "  PASS $1"
  else
    echo "  FAIL $1 (missing: $2)"
    failures=$((failures + 1))
  fi
}

CSS_FILE=$(find dist/_astro -name '*.css' 2>/dev/null | head -1)
check "daisyUI CSS"          'daisyui' "$CSS_FILE"
check "theme-change script"  'themeChange'
check "dark default theme"   'data-theme="dark"'
check "theme picker"         'data-choose-theme'
check "daisyUI card"         'card bg-base-100'
check "daisyUI btn"          'btn-primary'
check "daisyUI select"       'select-bordered'

if [ "$failures" -eq 0 ]; then
  echo "=== All $((7 - failures)) checks passed ==="
else
  echo "=== $failures check(s) FAILED ==="
  exit 1
fi
