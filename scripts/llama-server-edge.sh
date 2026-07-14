#!/usr/bin/env bash
# llama-server launch template for hermes-edge (Unix/macOS).
#
# Carries the server half of the edge contract (docs/edge/config.yaml):
# the agent pins its conversation to a slot via request cache_key; the
# server keeps per-slot context checkpoints and parks evicted states in
# host RAM so auxiliary traffic can never silently destroy the main
# conversation's prefix cache.
#
# Flag provenance:
#   --ctx-checkpoints   upstream llama.cpp (PR 15293) — SWA/hybrid restore points
#   --cache-ram         upstream llama.cpp (PR 16391) — parked states, checkpoint-inclusive
#   cache_key binding   fork feature (e.g. llama-cpp-turboquant); harmless if absent
#
# Everything marked CHANGEME is deployment-specific. Measure, don't guess:
#   prefill tok/s  -> feed into local_runtime.prefill_tps in the profile
#   state size     -> server log prints MiB per parked state; size --cache-ram
#   VRAM           -> nvidia-smi while at full context; adjust -ngl / KV quant

set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"   # CHANGEME: path to your build
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the .gguf file}"
ALIAS="${ALIAS:-CHANGEME-model-alias}"          # must match model.default in the profile
PORT="${PORT:-8080}"

# CPU split (example: Intel 6P+8E => decode on P-cores, prefill on all cores;
# decode is memory-bound — more threads than physical P-cores usually hurts):
THREADS_DECODE="${THREADS_DECODE:-6}"
THREADS_PREFILL="${THREADS_PREFILL:-14}"

# COEXIST=1 — leave ~half the machine for other work. Measured cost (M1 Pro,
# full-RAM CPU decode): halving threads cost decode −16% and IMPROVED prefill
# +15% (efficiency-core spill and cross-process contention disappear) —
# decode's wall is memory bandwidth, not cores. RAM already shares
# gracefully: weights are mmap'd and never mlocked, so the OS reclaims pages
# for other processes and inference degrades toward the measured paging
# floor instead of OOMing. For GPU sharing, lower -ngl until `nvidia-smi`
# shows the VRAM you want free — CUDA time-slices compute automatically.
if [ "${COEXIST:-0}" = "1" ]; then
  THREADS_DECODE=$(( THREADS_DECODE / 2 < 2 ? 2 : THREADS_DECODE / 2 ))
  THREADS_PREFILL=$(( THREADS_PREFILL / 2 < 2 ? 2 : THREADS_PREFILL / 2 ))
  NICE_LEVEL="${NICE_LEVEL:-15}"
fi

exec nice -n "${NICE_LEVEL:-5}" "$LLAMA_SERVER" \
  --model "$MODEL_PATH" \
  --alias "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" \
  --ctx-size 131072 \
  --parallel 2 \
  `# hermes model.context_length must be ctx-size / parallel (per-slot)` \
  --ctx-checkpoints 32 \
  --cache-ram 4096 \
  --threads "$THREADS_DECODE" \
  --threads-batch "$THREADS_PREFILL" \
  --flash-attn on \
  "$@"

# Model/GPU-specific flags to append via "$@" or here, per deployment:
#   -ngl 99 --n-cpu-moe 999          # MoE: attention on GPU, experts on CPU
#   -ctk q8_0 -ctv q8_0              # halve KV VRAM (needs flash-attn)
#   --slot-save-path /path/to/slots  # optional: persist slots across restarts
#   --ubatch-size 512                # prefill batch; tune against prefill tok/s
