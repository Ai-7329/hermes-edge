@echo off
rem llama-server launch template for hermes-edge (Windows).
rem See scripts/llama-server-edge.sh for flag provenance and sizing notes.
rem
rem /BELOWNORMAL keeps the agent's backend polite on a shared box: prefill
rem saturates all cores, and the box is expected to keep doing other work.
rem Pin cores instead with: start "" /AFFINITY 0x3FFF ... (mask = allowed CPUs)

setlocal

if "%LLAMA_SERVER%"=="" set LLAMA_SERVER=llama-server.exe
if "%MODEL_PATH%"==""   set MODEL_PATH=CHANGEME\model.gguf
if "%ALIAS%"==""        set ALIAS=CHANGEME-model-alias
if "%PORT%"==""         set PORT=8080

rem CPU split (example i5-14500, 6P+8E): decode on P-cores, prefill on all.
if "%THREADS_DECODE%"==""  set THREADS_DECODE=6
if "%THREADS_PREFILL%"=="" set THREADS_PREFILL=14

rem COEXIST=1 — leave ~half the machine for other work. Measured: halving
rem threads costs decode about -16% and can IMPROVE prefill (E-core spill and
rem contention disappear); decode's wall is memory bandwidth, not cores.
rem Pin to P-cores on i5-14500 with: start "" /BELOWNORMAL /AFFINITY 0xFFF ...
rem (logical CPUs 0-11 = 6 P-cores with HT). RAM shares gracefully (mmap, no
rem mlock). For GPU headroom, lower -ngl until nvidia-smi shows the free VRAM
rem you want; CUDA time-slices compute between processes automatically.
if "%COEXIST%"=="1" (
  set THREADS_DECODE=4
  set THREADS_PREFILL=8
)

start "llama-server-edge" /BELOWNORMAL "%LLAMA_SERVER%" ^
  --model "%MODEL_PATH%" ^
  --alias "%ALIAS%" ^
  --host 127.0.0.1 --port %PORT% ^
  --ctx-size 131072 ^
  --parallel 2 ^
  --ctx-checkpoints 32 ^
  --cache-ram 4096 ^
  --threads %THREADS_DECODE% ^
  --threads-batch %THREADS_PREFILL% ^
  --flash-attn on ^
  %*

rem Model/GPU-specific flags to append via %* or here, per deployment:
rem   -ngl 99 --n-cpu-moe 999          (MoE: attention on GPU, experts on CPU)
rem   -ctk q8_0 -ctv q8_0              (halve KV VRAM; needs flash-attn)
rem   --slot-save-path C:\hermes\slots (optional: persist slots across restarts)
rem   --ubatch-size 512                (prefill batch; tune against prefill tok/s)

endlocal
