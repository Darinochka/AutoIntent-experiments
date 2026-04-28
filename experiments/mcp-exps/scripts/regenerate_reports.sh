#!/usr/bin/env bash
# Regenerate every JSONL report under ./reports/ from Logfire (experiment name = filename stem).
# Uses a delay between runs to reduce Logfire API rate-limit hits.
#
# Usage (from repo mcp-exps/):
#   ./scripts/regenerate_reports.sh
#
# Optional env:
#   REGENERATE_REPORTS_DELAY_SEC  seconds between runs (default: 45)
#   REPORT_TIMEOUT_SEC           --timeout for Logfire client (default: 180)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

DELAY="${REGENERATE_REPORTS_DELAY_SEC:-45}"
TIMEOUT="${REPORT_TIMEOUT_SEC:-180}"

jsonl_files=()
while IFS= read -r f; do
  [[ -n "$f" ]] && jsonl_files+=("$f")
done < <(find reports -maxdepth 1 -name '*.jsonl' | LC_ALL=C sort)

if [[ ${#jsonl_files[@]} -eq 0 ]]; then
  echo "No reports/*.jsonl files found under ${ROOT}/reports"
  exit 1
fi

failed=()
n=0
total=${#jsonl_files[@]}

echo "Regenerating ${total} report(s); delay ${DELAY}s between runs; timeout ${TIMEOUT}s."
echo

for f in "${jsonl_files[@]}"; do
  n=$((n + 1))
  base="$(basename "$f" .jsonl)"
  echo "[${n}/${total}] ${base}"

  if uv run python report.py "${base}" --output-dir ./reports --timeout "${TIMEOUT}"; then
    echo "  OK"
  else
    echo "  FAILED"
    failed+=("${base}")
  fi

  if [[ "${n}" -lt "${total}" ]]; then
    echo "  sleeping ${DELAY}s..."
    sleep "${DELAY}"
  fi
  echo
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed (${#failed[@]}): ${failed[*]}"
  exit 1
fi

echo "All ${total} report(s) regenerated successfully."
exit 0
