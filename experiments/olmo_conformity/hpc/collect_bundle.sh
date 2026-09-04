#!/bin/bash
# Pack a finished TAG for download (login node). Each job already wrote its bundle/ pieces; this re-runs the
# cross-variant summary and tars bundle/ (parquets without reasoning text, probes, summaries, manifests, logs).
#   TAG=full_0905 ./collect_bundle.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
B="${AAM_BELIEF_DIR}/bundle"; mkdir -p "${B}"
python investigation/backstudy/analysis_belief_probe.py --data-dir "${AAM_BELIEF_DIR}" --out-dir "${B}" || true
cp -f "${AAM_BELIEF_DIR}"/logs/*.log "${B}/" 2>/dev/null || true
cp -f "${AAM_BELIEF_DIR}"/*.manifest.json "${B}/" 2>/dev/null || true
tar -czf "${AAM_BELIEF_DIR}/${TAG}_bundle.tar.gz" -C "${AAM_BELIEF_DIR}" bundle
echo "Bundle: ${AAM_BELIEF_DIR}/${TAG}_bundle.tar.gz ($(du -h "${AAM_BELIEF_DIR}/${TAG}_bundle.tar.gz" | cut -f1))"
echo "Raw jsonl + activations (large; for the external drive): ${AAM_BELIEF_DIR}/*.jsonl  ${AAM_BELIEF_DIR}/activations/"
