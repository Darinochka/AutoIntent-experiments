#!/usr/bin/env bash
# Build merged JSONL reports for every cross-validation block in README.md (tool-suggest OOS CV),
# via `report.py aggregate-links` (one output per model row).
#
# Usage (from repo mcp-exps/):
#   ./scripts/aggregate_cv_readme_reports.sh
#
# Requires LOGFIRE_API_KEY (e.g. in .env); `report.py` loads dotenv from cwd.
#
# Optional env:
#   AGGREGATE_CV_BETWEEN_MODELS_SEC  pause after each full 5-fold aggregate (default: 90; avoids Logfire per-min rate limits)
#   REPORT_TIMEOUT_SEC               --timeout for Logfire client (default: 300)
#   AGGREGATE_CV_INTER_LINK_DELAY     passed to --inter-link-delay (default: 45; each link runs 2 SQL queries)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p "${ROOT}/reports"

BETWEEN="${AGGREGATE_CV_BETWEEN_MODELS_SEC:-90}"
TIMEOUT="${REPORT_TIMEOUT_SEC:-300}"
INTER_LINK="${AGGREGATE_CV_INTER_LINK_DELAY:-45}"

failed=()
n=0
total=7

aggregate_cv() {
  local name=$1
  shift
  uv run report.py aggregate-links \
    --name "${name}" \
    --output "${ROOT}/reports/${name}.jsonl" \
    --timeout "${TIMEOUT}" \
    --inter-link-delay "${INTER_LINK}" \
    "$@"
}

echo "Writing under ${ROOT}/reports ; ${BETWEEN}s pause between models; Logfire timeout ${TIMEOUT}s."
echo

run_one() {
  local label=$1
  shift
  n=$((n + 1))
  echo "[${n}/${total}] ${label}"
  if aggregate_cv "${label}" "$@"; then
    echo "  OK"
  else
    echo "  FAILED"
    failed+=("${label}")
  fi
  if [[ "${n}" -lt "${total}" ]]; then
    echo "  sleeping ${BETWEEN}s..."
    sleep "${BETWEEN}"
  fi
  echo
}

# --- cross-validation (README: #### cross-validation) ---
run_one "cv-readme-haiku-4-5" \
  --link "https://logfire-eu.pydantic.dev/public-trace/a40ea1e0-a11f-4e25-ab99-fc8866d70e46?spanId=af8536a54892c45b" \
  --link "https://logfire-eu.pydantic.dev/public-trace/499bdc10-a7ec-4b0d-831d-817ec595a51b?spanId=844a931495fd6ec7" \
  --link "https://logfire-eu.pydantic.dev/public-trace/90378452-cbe7-41b1-911d-1622ec023b4d?spanId=516c4b3384b62286" \
  --link "https://logfire-eu.pydantic.dev/public-trace/d8134193-ee25-405a-90f9-4539f34a0044?spanId=9709bca9354a7097" \
  --link "https://logfire-eu.pydantic.dev/public-trace/23b2a39a-fc95-41b5-9fe1-3786193abe6c?spanId=356c8caab8ef368e"

run_one "cv-readme-opus-4-6" \
  --link "https://logfire-eu.pydantic.dev/public-trace/947b3160-0075-41a6-8022-bb142f780da7?spanId=d43e619e2351a715" \
  --link "https://logfire-eu.pydantic.dev/public-trace/5121a1c6-6c58-4ee3-8151-eedf399ee32d?spanId=f48245b230e40317" \
  --link "https://logfire-eu.pydantic.dev/public-trace/ffb9ec89-b39f-4d86-bd76-54359b870815?spanId=5f77602f762beb88" \
  --link "https://logfire-eu.pydantic.dev/public-trace/92292e1e-5d09-40e4-a64f-da74f89e87bd?spanId=4a2e7e6aaff73195" \
  --link "https://logfire-eu.pydantic.dev/public-trace/0336e1ec-cc81-42f7-ae9d-cc5f96c1a204?spanId=c5bb4d5d5330587f"

run_one "cv-readme-gpt-5-4" \
  --link "https://logfire-eu.pydantic.dev/public-trace/476f9505-0956-43d1-a5e0-3872694ab88a?spanId=9104dcf21ca0852b" \
  --link "https://logfire-eu.pydantic.dev/public-trace/2c9a4990-956d-40c7-a602-43b8a2f87ceb?spanId=963c1b188cf0f601" \
  --link "https://logfire-eu.pydantic.dev/public-trace/dd0cabb1-801d-41d3-9a59-032a699102b2?spanId=5fbb812772da006c" \
  --link "https://logfire-eu.pydantic.dev/public-trace/7df0543e-745e-4975-b50e-fc22c8b53bad?spanId=798e4f34c923901e" \
  --link "https://logfire-eu.pydantic.dev/public-trace/4fdf9b34-6452-466e-afca-07799489e862?spanId=7d60455f96ff1f4a"

# Output stem matches `README_BASIC_VS_CV` (compare-readme)
run_one "cv-gpt54-mini-aggregated" \
  --link "https://logfire-eu.pydantic.dev/public-trace/f96a957b-5db1-4972-9397-018053aa1857?spanId=9867a54a3fcd4f2a" \
  --link "https://logfire-eu.pydantic.dev/public-trace/45335ca7-cb4a-485f-9684-2afedcbeefce?spanId=73deee9365c25f6f" \
  --link "https://logfire-eu.pydantic.dev/public-trace/9f8ae371-b171-4142-9796-d4ba2188fc15?spanId=df097ffe8202c923" \
  --link "https://logfire-eu.pydantic.dev/public-trace/5043d59a-eb2c-4111-bad3-ef4635ab0f01?spanId=f8f0da3bfb034bd3" \
  --link "https://logfire-eu.pydantic.dev/public-trace/cfa2ebb4-713a-4902-b348-1b7dd8d2d1bd?spanId=5fba0445fbc07824"

run_one "cv-readme-gpt-5-4-nano" \
  --link "https://logfire-eu.pydantic.dev/public-trace/06ab1d36-fc75-4d06-9908-f4656ff5c64c?spanId=1ca6d9e816cdd3cf" \
  --link "https://logfire-eu.pydantic.dev/public-trace/31c96cec-e9c2-495d-978b-ac2b5eb5c45e?spanId=df72fbed93161f22" \
  --link "https://logfire-eu.pydantic.dev/public-trace/c530eb72-2ace-4a78-b80d-350f0605d5e8?spanId=332d29a7165246ef" \
  --link "https://logfire-eu.pydantic.dev/public-trace/1602d850-0dd4-4a6a-b3dc-e24243ca2c67?spanId=51c53a63b8b21333" \
  --link "https://logfire-eu.pydantic.dev/public-trace/0ad2dbd6-2acb-462d-8e88-204f8d4df41e?spanId=f3e7fec0c5c6ea60"

run_one "cv-readme-qwen3-coder-plus" \
  --link "https://logfire-eu.pydantic.dev/public-trace/db27f960-6075-44cb-a88e-2e9e3f0f3041?spanId=4e31ab5347bba8e7" \
  --link "https://logfire-eu.pydantic.dev/public-trace/1ed1cde4-9a4d-4d62-860d-cc8a0930ebae?spanId=3859a151d6aeeb03" \
  --link "https://logfire-eu.pydantic.dev/public-trace/9ff1129f-3a99-4026-b834-02067fcf2776?spanId=0c6efc7c159ed54c" \
  --link "https://logfire-eu.pydantic.dev/public-trace/d0ce7cfd-15ad-4b70-a5ad-e2cfcab128df?spanId=7c774a1cab692ab5" \
  --link "https://logfire-eu.pydantic.dev/public-trace/ce805346-1f1a-41b6-a35b-8b3c06838acd?spanId=790b0905f01fd1a0"

run_one "cv-readme-deepseek-v3-2" \
  --link "https://logfire-eu.pydantic.dev/public-trace/ba7d18b9-3574-4900-b870-4a2755167651?spanId=22005625f27fb358" \
  --link "https://logfire-eu.pydantic.dev/public-trace/61eda002-a80f-49c9-b149-8197813c26ba?spanId=bc71e6478302978a" \
  --link "https://logfire-eu.pydantic.dev/public-trace/9b7668e2-5e3b-4e8a-8941-008a38cf2638?spanId=beb2578330b4f46b" \
  --link "https://logfire-eu.pydantic.dev/public-trace/551044ed-67bf-41ce-99fc-dbfb713de103?spanId=2b98d3e96a14f1ab" \
  --link "https://logfire-eu.pydantic.dev/public-trace/c5a5f727-a8a4-45a6-a3cd-7fb30990f20a?spanId=b0c8914a54606ac8"

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed (${#failed[@]}): ${failed[*]}"
  exit 1
fi

echo "All ${total} CV aggregate report(s) finished successfully."
exit 0
