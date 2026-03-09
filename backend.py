"""
ENGRAM Quantum Control Interface — Backend
==========================================
Quantis USB-4M  →  pyqpanda Floquet circuit  →  H100 GPU  →  IPR / SSE stream

Hardware path:
  1. Quantis USB-4M supplies true random bytes via libQuantis.dll
  2. Those bytes parameterise a d-mode Floquet DTC circuit (RX/RZ rotations)
  3. The circuit runs on the H100 via pyqpanda's GPUQVM (CUDA-Q backend)
  4. Resulting state vector → IPR calculation → SSE stream to frontend

Fallbacks (for testing without hardware):
  - No Quantis  → numpy PRNG
  - No GPU/CUDA → pyqpanda CPUQVM
  - No pyqpanda → pure numpy state evolution

Requirements:
    pip install fastapi uvicorn numpy pyqpanda ollama

Run:
    uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import ctypes
import json
import math
import time
import threading
from collections import deque
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# ── pyqpanda ──────────────────────────────────────────────────────────────────
# We try the GPU VM first (H100), fall back to CPU VM, then to pure numpy.
PYQPANDA_AVAILABLE = False
GPU_AVAILABLE = False

try:
    import pyqpanda as pq
    PYQPANDA_AVAILABLE = True

    # Try GPU backend — requires CUDA-Q / Origin GPU driver on H100
    try:
        _test_vm = pq.GPUQVM()
        _test_vm.init_qvm()
        _test_vm.finalize()
        GPU_AVAILABLE = True
        print("[OK] pyqpanda GPUQVM (H100) available")
    except Exception as _gpu_err:
        print(f"[WARN] GPUQVM not available ({_gpu_err}) — using CPUQVM")

except ImportError:
    print("[WARN] pyqpanda not found — running numpy fallback")

# ── Ollama ────────────────────────────────────────────────────────────────────
try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    _ollama = None
    OLLAMA_AVAILABLE = False
    print("[WARN] ollama not found — chat returns stub responses")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="ENGRAM Quantum Control API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 1.  QUANTIS ENTROPY SOURCE
# =============================================================================
_quantis_lib = None
QUANTIS_AVAILABLE = False

# libQuantis.dll ships with the ID Quantique driver package.
_QUANTIS_DLL_PATHS = [
    "libQuantis.dll",
    r"C:\Program Files\ID Quantique\Quantis\libQuantis.dll",
    r"C:\Program Files (x86)\ID Quantique\Quantis\libQuantis.dll",
]

for _path in _QUANTIS_DLL_PATHS:
    try:
        _quantis_lib = ctypes.CDLL(_path)
        _rc = _quantis_lib.QuantisOpen(0)   # device index 0
        if _rc == 0:
            QUANTIS_AVAILABLE = True
            print(f"[OK] Quantis USB-4M found via {_path}")
            break
        else:
            _quantis_lib = None
    except Exception:
        _quantis_lib = None

if not QUANTIS_AVAILABLE:
    print("[WARN] Quantis not detected — using numpy PRNG as entropy source")


def read_quantis_bytes(n_bytes: int) -> np.ndarray:
    """
    Return n_bytes of entropy as uint8 array.
    Uses Quantis hardware if available, otherwise numpy PRNG.
    """
    if _quantis_lib and QUANTIS_AVAILABLE:
        buf = (ctypes.c_ubyte * n_bytes)()
        # QuantisRead(device_type=1 for USB, device_number=0, buffer, size)
        ret = _quantis_lib.QuantisRead(
            ctypes.c_int(1), ctypes.c_int(0),
            buf, ctypes.c_size_t(n_bytes)
        )
        if ret == 0:
            return np.frombuffer(bytes(buf), dtype=np.uint8)
    # Fallback
    return (np.random.random(n_bytes) * 255).astype(np.uint8)


def entropy_angles(n_modes: int) -> np.ndarray:
    """
    Convert Quantis bytes into rotation angles in [0, 2pi).
    Each angle uses 4 bytes for 32-bit float resolution.
    """
    raw = read_quantis_bytes(n_modes * 4)
    # Pad if needed (can happen on first read)
    if len(raw) < n_modes * 4:
        raw = np.concatenate([raw, np.zeros(n_modes * 4 - len(raw), dtype=np.uint8)])
    uint32s = raw.view(np.uint32)[:n_modes]
    return (uint32s.astype(np.float64) / float(0xFFFFFFFF)) * 2.0 * np.pi


# =============================================================================
# 2.  RIEMANN ZETA FREQUENCY COMB  (drives the Floquet periods)
# =============================================================================
# First 20 non-trivial zeros of zeta(s).
# Extend toward 100 zeros for the full d-100 simulation described in the paper.
ZETA_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
]

N_MODES = len(ZETA_ZEROS)   # 20 right now; matches number of qubits


def floquet_period(mode_index: int, t: float) -> float:
    """
    Floquet driving period for mode i: T_i = 2pi / gamma_i
    Each mode oscillates at its own Zeta frequency.
    """
    gamma = ZETA_ZEROS[mode_index % len(ZETA_ZEROS)]
    return (2.0 * math.pi / gamma) * (1.0 + 0.01 * math.sin(t))


# =============================================================================
# 3.  PYQPANDA FLOQUET CIRCUIT  (Quantis entropy → H100 via GPUQVM)
# =============================================================================

def run_floquet_circuit(angles: np.ndarray, t: float) -> np.ndarray:
    """
    Build and run one Floquet DTC step.

    Circuit per qubit i:
        RX(angles[i])       — inject Quantis entropy as rotation
        RZ(T_i(t))          — Floquet drive at Zeta-comb frequency
        CNOT(i, i+1)        — nearest-neighbour entanglement (MBL coupling)

    Returns probability distribution over 2^N_MODES basis states.
    Falls back to numpy if pyqpanda unavailable.
    """
    if PYQPANDA_AVAILABLE:
        try:
            vm = pq.GPUQVM() if GPU_AVAILABLE else pq.CPUQVM()
            vm.init_qvm()
            qubits = vm.qAlloc_many(N_MODES)

            prog = pq.QProg()

            # Entropy injection + Floquet drive
            for i in range(N_MODES):
                prog << pq.RX(qubits[i], float(angles[i]))
                prog << pq.RZ(qubits[i], floquet_period(i, t))

            # Nearest-neighbour entanglement (MBL coupling)
            for i in range(N_MODES - 1):
                prog << pq.CNOT(qubits[i], qubits[i + 1])

            result = vm.prob_run_list(prog, qubits, -1)
            vm.finalize()

            state = np.abs(np.array(result, dtype=np.float64))
            total = state.sum()
            return state / total if total > 0 else state

        except Exception as e:
            print(f"[WARN] pyqpanda circuit error: {e} — numpy fallback")

    # ── Numpy fallback ────────────────────────────────────────────────────────
    # Tensor product state evolved under Floquet phases
    single = np.array([math.cos(angles[0] / 2.0), math.sin(angles[0] / 2.0)])
    state = single
    for i in range(1, N_MODES):
        q = np.array([math.cos(angles[i] / 2.0), math.sin(angles[i] / 2.0)])
        state = np.kron(state, q)

    # Apply Floquet phase modulation
    phases = np.array([floquet_period(i, t) for i in range(N_MODES)])
    total_phase = float(np.sum(phases))
    idx = np.linspace(0.0, 1.0, len(state))
    state = np.abs(state * np.exp(1j * total_phase * idx))
    total = state.sum()
    return state / total if total > 0 else state


# =============================================================================
# 4.  IPR — INVERSE PARTICIPATION RATIO
# =============================================================================

def calculate_ipr(prob_dist: np.ndarray) -> float:
    """
    IPR = 1 / sum(p_i^2)

    Low IPR  → delocalised → many basis states occupied → SYNTROPY LOCK
    High IPR → localised   → few basis states occupied  → ENTROPY DRIFT

    Paper thresholds: <0.3 locked, 0.3–0.6 marginal, >0.6 drift
    """
    p = np.clip(prob_dist, 1e-15, None)
    p = p / p.sum()
    return float(1.0 / np.sum(p ** 2))


# =============================================================================
# 5.  ENGRAM MEMORY ROUTER
# =============================================================================

class EngramRouter:
    IPR_LOCK  = 0.3
    IPR_DRIFT = 0.6

    def __init__(self):
        self.S: float     = 0.0
        self.Sigma: float = 1.0
        self.episodic: deque = deque(maxlen=500)

    def route(self, ipr: float) -> str:
        return "deepseek-v3" if ipr >= self.IPR_LOCK else "llama3.3"

    def memory_stores(self, ipr: float) -> list:
        stores = ["episodic"]
        if ipr > self.IPR_LOCK:
            stores.append("semantic")
        if ipr > self.IPR_DRIFT:
            stores.append("procedural")
        return stores

    def log(self, record: dict):
        self.episodic.append(record)


# =============================================================================
# 6.  SHARED SIMULATION STATE
# =============================================================================

class SimState:
    def __init__(self):
        self.running           = False
        self.cycle             = 0
        self.ipr               = 0.18
        self.entropy_val       = 0.0
        self.syntropy_val      = 1.0
        self.ds_squared        = 0.0
        self.fidelity          = 1.0
        self.controller        = "llama3.3"
        self.memory_stores_val = ["episodic"]
        self.status            = "IDLE"
        self.correction        = False
        self.history: deque    = deque(maxlen=500)
        self.lock              = threading.Lock()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "cycle":                self.cycle,
                "ipr":                  round(self.ipr, 4),
                "entropy":              round(self.entropy_val, 4),
                "syntropy":             round(self.syntropy_val, 4),
                "ds_squared":           round(self.ds_squared, 6),
                "fidelity":             round(self.fidelity, 6),
                "controller":           self.controller,
                "memory_stores":        self.memory_stores_val,
                "status":               self.status,
                "correction_triggered": self.correction,
                "quantis_active":       QUANTIS_AVAILABLE,
                "gpu_active":           GPU_AVAILABLE,
                "history":              list(self.history),
            }


sim    = SimState()
router = EngramRouter()


# =============================================================================
# 7.  MAIN SIMULATION LOOP
# =============================================================================

def _simulation_loop():
    """
    Runs in a background thread.

    Each iteration:
      A) Read Quantis entropy → rotation angles for each qubit
      B) Run Floquet circuit on H100 (or fallback) → probability distribution
      C) Calculate IPR from resulting distribution
      D) Compute entropy S and syntropy Sigma
      E) Route to controller, update shared state, stream to frontend
    """
    cycle = 0
    t     = 0.0

    while sim.running:
        t0 = time.time()

        # A: Entropy from Quantis (or PRNG fallback)
        angles = entropy_angles(N_MODES)

        # B: Floquet circuit on H100 (or fallback)
        prob_dist = run_floquet_circuit(angles, t)

        # C: IPR — normalised to [0, 1] display range
        raw_ipr = calculate_ipr(prob_dist)
        ipr = float(np.clip(raw_ipr / max(len(prob_dist), 1), 0.05, 1.0))

        # D: Entropy S (von Neumann) and Syntropy Sigma
        p = np.clip(prob_dist, 1e-15, None)
        p = p / p.sum()
        new_S     = float(-np.sum(p * np.log(p)))
        new_Sigma = float(np.sum(np.sqrt(p * np.roll(p, 1))))
        dS        = new_S     - router.S
        dSigma    = new_Sigma - router.Sigma
        ds_sq     = dS ** 2 - dSigma ** 2
        router.S      = new_S
        router.Sigma  = new_Sigma

        # E: Controller routing
        controller = router.route(ipr)
        mem_stores = router.memory_stores(ipr)
        fidelity   = float(np.clip(1.0 - abs(ds_sq), 0.0, 1.0))
        correction = (ipr > router.IPR_DRIFT)

        if ipr < router.IPR_LOCK:
            status = "SYNTROPY LOCK"
        elif ipr < router.IPR_DRIFT:
            status = "MARGINAL"
        else:
            status = "ENTROPY DRIFT"

        record = {
            "cycle":      cycle,
            "ipr":        round(ipr, 4),
            "entropy":    round(new_S, 4),
            "syntropy":   round(new_Sigma, 4),
            "fidelity":   round(fidelity, 6),
            "ds_squared": round(ds_sq, 6),
        }
        router.log(record)

        with sim.lock:
            sim.cycle              = cycle
            sim.ipr                = ipr
            sim.entropy_val        = new_S
            sim.syntropy_val       = new_Sigma
            sim.ds_squared         = ds_sq
            sim.fidelity           = fidelity
            sim.controller         = controller
            sim.memory_stores_val  = mem_stores
            sim.status             = status
            sim.correction         = correction
            sim.history.append(record)

        cycle += 1
        t     += 0.1

        elapsed = time.time() - t0
        time.sleep(max(0.0, 0.5 - elapsed))   # ~2 cycles/sec


# =============================================================================
# 8.  API ROUTES
# =============================================================================

@app.get("/health")
def health():
    return {
        "quantis":            QUANTIS_AVAILABLE,
        "pyqpanda":           PYQPANDA_AVAILABLE,
        "gpu":                GPU_AVAILABLE,
        "ollama":             OLLAMA_AVAILABLE,
        "simulation_running": sim.running,
        "entropy_source":     "QRNG (Quantis USB-4M)" if QUANTIS_AVAILABLE else "PRNG (numpy fallback)",
        "compute_backend":    "H100 GPUQVM" if GPU_AVAILABLE else ("CPUQVM" if PYQPANDA_AVAILABLE else "numpy"),
    }


@app.get("/status")
def get_status():
    return JSONResponse(sim.snapshot())


@app.get("/stream")
async def stream(request: Request):
    """SSE stream — emits a JSON event every 500ms."""
    async def gen() -> AsyncGenerator[str, None]:
        while True:
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps(sim.snapshot())}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/start")
def start():
    if sim.running:
        return {"status": "already_running"}
    sim.running = True
    threading.Thread(target=_simulation_loop, daemon=True).start()
    return {
        "status":          "started",
        "entropy_source":  "QRNG" if QUANTIS_AVAILABLE else "PRNG",
        "compute_backend": "H100 GPUQVM" if GPU_AVAILABLE else ("CPUQVM" if PYQPANDA_AVAILABLE else "numpy"),
    }


@app.post("/stop")
def stop():
    sim.running = False
    return {"status": "stopped"}


@app.get("/history")
def history():
    with sim.lock:
        return JSONResponse({"history": list(sim.history)})


@app.post("/chat")
async def chat(request: Request):
    """Query DeepSeek-V3 or Llama 3.3 via Ollama with live sim state injected."""
    body    = await request.json()
    message = body.get("message", "").strip()
    model   = body.get("model", sim.controller)

    if not message:
        return JSONResponse({"error": "empty message"}, status_code=400)

    snap = sim.snapshot()
    system_prompt = (
        f"You are an autonomous quantum control agent for the ENGRAM system.\n"
        f"Entropy source: {'Quantis USB-4M (true QRNG)' if QUANTIS_AVAILABLE else 'PRNG fallback'}\n"
        f"Compute backend: {'H100 GPUQVM' if GPU_AVAILABLE else 'CPU'}\n"
        f"Cycle: {snap['cycle']} | IPR: {snap['ipr']} | Status: {snap['status']}\n"
        f"Entropy S: {snap['entropy']} | Syntropy Sigma: {snap['syntropy']}\n"
        f"Fidelity: {snap['fidelity']} | ds_squared: {snap['ds_squared']}\n"
        f"Active memory stores: {', '.join(snap['memory_stores'])}\n"
        f"Respond concisely. Reference live values where relevant."
    )

    if OLLAMA_AVAILABLE and _ollama:
        try:
            resp  = _ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": message},
                ],
            )
            reply = resp["message"]["content"]
        except Exception as e:
            reply = f"[Ollama error: {e}]"
    else:
        reply = (
            f"[Ollama not connected] "
            f"Cycle {snap['cycle']}, IPR={snap['ipr']}, "
            f"status={snap['status']}, fidelity={snap['fidelity']}."
        )

    return JSONResponse({
        "model":         model,
        "reply":         reply,
        "ipr":           snap["ipr"],
        "status":        snap["status"],
        "memory_stores": snap["memory_stores"],
    })