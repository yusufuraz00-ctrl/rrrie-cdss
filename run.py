"""RRRIE-CDSS Unified Launcher — tek komutla her şeyi başlat.

Kullanım:
    py run.py                  # llama-server + GUI başlat
    py run.py --test           # llama-server + 3 varsayılan test çalıştır
    py run.py --test --all     # llama-server + tüm 23+ test
    py run.py --test --narrative --count 5   # 5 narrative test
    py run.py --test --who     # sadece WHO vakaları test et
    py run.py --gui            # sadece GUI başlat (server zaten açıksa)
    py run.py --status         # sistem durumunu kontrol et
    py run.py --stop           # llama-server'ı durdur

Otomatik olarak:
  ✓ GGUF model dosyasını bulur
  ✓ llama-server'ı başlatır ve hazır olmasını bekler
  ✓ GUI veya test pipeline'ını çalıştırır
"""

from __future__ import annotations

import argparse
import glob
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
LLAMA_SERVER_EXE = PROJECT_ROOT / "llama-server" / "llama-server.exe"
ENV_FILE = PROJECT_ROOT / ".env"

# llama-server defaults
LLAMA_HOST = "127.0.0.1"
MAIN_LLAMA_PORT = 8080       # Port for 4B diagnosis model
DLLM_LLAMA_PORT = 8081       # Port for 0.8B R0 reasoning engine
GPU_LAYERS = 99
CTX_SIZE = 8192  # 8K is sufficient — saves VRAM for faster generation

# ── ANSI Colors ─────────────────────────────────────────────────────
G = "\033[92m"  # green
Y = "\033[93m"  # yellow
R = "\033[91m"  # red
C = "\033[96m"  # cyan
B = "\033[1m"   # bold
RST = "\033[0m" # reset


def _print(icon: str, msg: str, color: str = ""):
    print(f"  {color}{icon}{RST} {msg}")


# ═══════════════════════════════════════════════════════════════════
# MODEL DISCOVERY — GGUF dosyasını otomatik bul
# ═══════════════════════════════════════════════════════════════════

# Search order: explicit env var → HF cache (Qwen3.5-4B) → HF cache (Qwen3-4B) → RRRIE/data/models
GGUF_SEARCH_PATTERNS = [
    # HuggingFace cache — blobs contain the actual data
    str(Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/blobs/*"),
    str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-GGUF/blobs/*"),
    # Project-local models
    str(PROJECT_ROOT / "models" / "*4B*.gguf"),
    str(PROJECT_ROOT / "models" / "*4b*.gguf"),
    str(PROJECT_ROOT.parent / "RRRIE" / "data" / "models" / "*.gguf"),
]

DLLM_SEARCH_PATTERNS = [
    str(PROJECT_ROOT / "models" / "*0.8B*.gguf"),
    str(PROJECT_ROOT / "models" / "*0.5B*.gguf"),
    str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/blobs/*"),
]

def find_gguf_model() -> tuple[Path | None, Path | None]:
    """Find usable GGUF model files automatically. Returns (main_model, dllm_model)."""
    main_p, dllm_p = None, None

    # 1. Main model (4B+)
    env_path = os.environ.get("GGUF_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists() and p.stat().st_size > 100_000_000:
            main_p = p

    if not main_p:
        for pattern in GGUF_SEARCH_PATTERNS:
            for match in glob.glob(pattern):
                p = Path(match)
                if p.is_file() and p.stat().st_size > 100_000_000:
                    main_p = p
                    break
            if main_p: break

    # 2. DLLM model (0.8B / 0.5B)
    for pattern in DLLM_SEARCH_PATTERNS:
        for match in glob.glob(pattern):
            p = Path(match)
            if p.is_file() and p.stat().st_size > 100_000_000:
                dllm_p = p
                break
        if dllm_p: break

    return main_p, dllm_p



# ═══════════════════════════════════════════════════════════════════
# LLAMA-SERVER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

def is_server_running(host: str, port: int) -> bool:
    """TCP check — is something listening on the port?"""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (ConnectionRefusedError, OSError, TimeoutError):
        return False


def is_server_healthy(host: str, port: int) -> bool:
    """HTTP health check — is llama-server loaded and ready?"""
    try:
        import requests
        r = requests.get(f"http://{host}:{port}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def start_llama_server(model_path: Path, port: int, is_dllm: bool = False) -> subprocess.Popen:
    """Start llama-server as a background process with optimized flags."""
    if not LLAMA_SERVER_EXE.exists():
        _print("✗", f"llama-server.exe bulunamadı: {LLAMA_SERVER_EXE}", R)
        sys.exit(1)

    ctx = 2048 if is_dllm else CTX_SIZE
    
    cmd = [
        str(LLAMA_SERVER_EXE),
        "-m", str(model_path),
        "--host", LLAMA_HOST,
        "--port", str(port),
        "-ngl", str(GPU_LAYERS),
        "-c", str(ctx),
        # ── Performance flags ──
        "--flash-attn", "on",        # Flash Attention
        "-b", "512",                 # Batch size for prompt processing
        "--cache-type-k", "q8_0",    # KV cache K quantization
        "--cache-type-v", "q8_0",    # KV cache V quantization
        "--cache-prompt",            # Prompt caching
    ]

    label = "DLLM" if is_dllm else "Main"
    _print("⟳", f"{label} llama-server başlatılıyor [{port}]...", Y)
    _print(" ", f"  Model: {model_path.name}", C)

    # Start as detached process
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    # Wait until healthy (max 120s)
    for i in range(120):
        if proc.poll() is not None:
            # Process exited — read output
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            _print("✗", f"{label} llama-server başlatılamadı (exit code {proc.returncode})", R)
            # Print last 10 lines of output
            for line in out.strip().split("\n")[-10:]:
                _print(" ", line)
            sys.exit(1)

        if is_server_healthy(LLAMA_HOST, port):
            _print("✓", f"{label} llama-server hazır! ({i+1}s)", G)
            return proc

        time.sleep(1)
        if (i + 1) % 10 == 0:
            _print("⟳", f"  [{port}] Bekleniyor... {i+1}s", Y)

    _print("✗", f"{label} llama-server 120 saniyede hazır olamadı!", R)
    proc.terminate()
    sys.exit(1)


def stop_llama_server():
    """Kill any llama-server process."""
    if sys.platform == "win32":
        os.system("taskkill /f /im llama-server.exe >nul 2>&1")
    else:
        os.system("pkill -f llama-server")
    _print("✓", "llama-server durduruldu.", G)


# ═══════════════════════════════════════════════════════════════════
# STATUS CHECK
# ═══════════════════════════════════════════════════════════════════

def print_status():
    """Print system status overview."""
    print(f"\n{B}  RRRIE-CDSS Sistem Durumu{RST}")
    print("  " + "─" * 45)

    # Main llama-server
    if is_server_healthy(LLAMA_HOST, MAIN_LLAMA_PORT):
        _print("✓", f"Main Server: {G}ÇALIŞIYOR{RST} (http://{LLAMA_HOST}:{MAIN_LLAMA_PORT})")
    elif is_server_running(LLAMA_HOST, MAIN_LLAMA_PORT):
        _print("⚠", f"Main Server: {Y}PORT AÇIK ama sağlıksız{RST}")
    else:
        _print("✗", f"Main Server: {R}KAPALI{RST}")

    # DLLM llama-server
    if is_server_healthy(LLAMA_HOST, DLLM_LLAMA_PORT):
        _print("✓", f"DLLM Server: {G}ÇALIŞIYOR{RST} (http://{LLAMA_HOST}:{DLLM_LLAMA_PORT})")
    elif is_server_running(LLAMA_HOST, DLLM_LLAMA_PORT):
        _print("⚠", f"DLLM Server: {Y}PORT AÇIK ama sağlıksız{RST}")
    else:
        _print("✗", f"DLLM Server: {R}KAPALI{RST}")

    # GUI
    gui_running = False
    try:
        with socket.create_connection(("127.0.0.1", 7860), timeout=2):
            gui_running = True
    except Exception:
        pass

    if gui_running:
        _print("✓", f"GUI: {G}ÇALIŞIYOR{RST} (http://localhost:7860)")
    else:
        _print("✗", f"GUI: {R}KAPALI{RST}")

    # Models
    main_mod, dllm_mod = find_gguf_model()
    if main_mod:
        _print("✓", f"Main Model : {G}{main_mod.name}{RST} ({main_mod.stat().st_size / (1024**3):.1f} GB)")
    else:
        _print("✗", f"Main Model : {R}Bulunamadı{RST}")
        
    if dllm_mod:
        _print("✓", f"DLLM Model : {G}{dllm_mod.name}{RST} ({dllm_mod.stat().st_size / (1024**3):.1f} GB)")
    else:
        _print("✗", f"DLLM Model : {R}Bulunamadı{RST}")

    # .env
    if ENV_FILE.exists():
        _print("✓", f".env: {G}Mevcut{RST}")
    else:
        _print("✗", f".env: {R}Eksik{RST}")

    print()


# ═══════════════════════════════════════════════════════════════════
# GUI LAUNCHER
# ═══════════════════════════════════════════════════════════════════

def launch_gui():
    """Launch the GUI server in foreground."""
    _print("✓", f"GUI başlatılıyor → {C}http://localhost:7860{RST}")
    print()

    # Change to project root so relative imports work
    os.chdir(str(PROJECT_ROOT))

    # Add project root to path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from gui.server import main as gui_main
    gui_main()


# ═══════════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_tests(extra_args: list[str]):
    """Run the e2e pipeline test with given arguments."""
    os.chdir(str(PROJECT_ROOT))

    test_script = PROJECT_ROOT / "tests" / "test_e2e_medical.py"
    cmd = [sys.executable, str(test_script)] + extra_args

    _print("▶", f"Pipeline testi başlatılıyor...", C)
    if extra_args:
        _print(" ", f"  Argümanlar: {' '.join(extra_args)}")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


# ═══════════════════════════════════════════════════════════════════
# MAIN — TEK GİRİŞ NOKTASI
# ═══════════════════════════════════════════════════════════════════

def ensure_servers() -> tuple[subprocess.Popen | None, subprocess.Popen | None]:
    """Ensure llama-servers are running. Start them if not."""
    main_proc = None
    dllm_proc = None

    _print("ℹ", f"Sistem: {sys.platform} | Python: {sys.version.split()[0]}")

    # ----- MAIN LLAMA-SERVER -----
    if not is_server_healthy(LLAMA_HOST, MAIN_LLAMA_PORT):
        main_model, dllm_model_for_warning = find_gguf_model() # Get dllm_model for warning
        if not main_model:
            _print("✗", "Main GGUF modeli bulunamadı!", R)
            _print(" ", "Çözüm: GGUF_MODEL_PATH environment variable ile yol belirtin:")
            _print(" ", "  set GGUF_MODEL_PATH=C:\\path\\to\\model.gguf")
            _print(" ", "  py run.py")
            sys.exit(1)
        
        # DLLM uyarısı (if main model is found but DLLM isn't)
        if not dllm_model_for_warning:
            _print("⚠", "DLLM modeli (~0.8B) bulunamadı! R0 fallback kullanacak.", Y)

        main_proc = start_llama_server(main_model, MAIN_LLAMA_PORT)
    else:
        _print("✓", f"Main llama-server zaten çalışıyor.", G)

    # ----- DLLM LLAMA-SERVER -----
    if not is_server_healthy(LLAMA_HOST, DLLM_LLAMA_PORT):
        _, dllm_model = find_gguf_model() # Re-find to ensure we have the DLLM model path
        if dllm_model:
            dllm_proc = start_llama_server(dllm_model, DLLM_LLAMA_PORT, is_dllm=True)
        else:
            _print("⚠", "DLLM (Preprocess) modeli yok. DLLM R0 pas geçilecek.", Y)
    else:
        _print("✓", f"DLLM llama-server zaten çalışıyor.", G)

    return main_proc, dllm_proc


def main():
    parser = argparse.ArgumentParser(
        description="RRRIE-CDSS — Tek komutla her şeyi başlat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  py run.py                     → llama-server + GUI başlat
  py run.py --test              → 3 varsayılan vaka test et
  py run.py --test --all        → tüm 23+ vakayı test et
  py run.py --test --narrative  → narrative vakaları test et
  py run.py --test --who        → WHO vakalarını test et
  py run.py --status            → sistem durumunu göster
  py run.py --stop              → llama-server'ı durdur
""",
    )

    parser.add_argument("--gui", action="store_true",
                        help="Sadece GUI başlat (llama-server zaten çalışıyorsa)")
    parser.add_argument("--test", action="store_true",
                        help="Pipeline testi çalıştır (GUI yerine)")
    parser.add_argument("--status", action="store_true",
                        help="Sistem durumunu göster")
    parser.add_argument("--stop", action="store_true",
                        help="llama-server'ı durdur")

    # Test options (passed through to test_e2e_medical.py)
    parser.add_argument("--all", action="store_true",
                        help="Tüm vakaları test et")
    parser.add_argument("--case", type=str,
                        help="Spesifik vaka testi (ör: pneumonia)")
    parser.add_argument("--narrative", action="store_true",
                        help="Narrative vakaları test et")
    parser.add_argument("--count", type=int, default=3,
                        help="Narrative vaka sayısı")
    parser.add_argument("--hard", action="store_true",
                        help="Sadece zor narrative vakalar")
    parser.add_argument("--mix", action="store_true",
                        help="Karışık mod: structured + narrative")
    parser.add_argument("--who", action="store_true",
                        help="Sadece WHO vakalarını test et")
    parser.add_argument("--list", action="store_true",
                        help="Mevcut vakaları listele")

    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────
    print(f"""
{B}╔══════════════════════════════════════════════════════════╗
║           RRRIE-CDSS — Unified Launcher                  ║
║     Clinical Decision Support System v2.0                ║
╚══════════════════════════════════════════════════════════╝{RST}
""")

    # ── Status ────────────────────────────────────────────────────
    if args.status:
        print_status()
        return

    # ── Stop ──────────────────────────────────────────────────────
    if args.stop:
        stop_llama_server()
        return

    # ── List ──────────────────────────────────────────────────────
    if args.list:
        # Pass --list to test script
        os.chdir(str(PROJECT_ROOT))
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from tests.narrative_cases import list_all_cases, NARRATIVE_CASES
        from tests.test_e2e_medical import ALL_CASES
        print(f"  {B}Structured Cases ({len(ALL_CASES)}):{RST}")
        for f in ALL_CASES:
            print(f"    • {f}")
        print(f"\n  {B}Narrative Cases ({len(NARRATIVE_CASES)}):{RST}")
        print(list_all_cases())
        print()
        return

    # ── Ensure llama-servers running ──────────────────────────────
    main_proc, dllm_proc = ensure_servers()

    try:
        if args.test or args.who:
            # ── Test Mode ────────────────────────────────────────
            test_args = []
            if args.who:
                # WHO cases are the who_*.json files
                test_args.extend(["--case", "who_"])
                if args.all:
                    test_args.append("--all")
            else:
                if args.all:
                    test_args.append("--all")
                if args.case:
                    test_args.extend(["--case", args.case])
                if args.narrative:
                    test_args.append("--narrative")
                if args.count != 3:
                    test_args.extend(["--count", str(args.count)])
                if args.hard:
                    test_args.append("--hard")
                if args.mix:
                    test_args.append("--mix")

            rc = run_tests(test_args)
            if rc == 0:
                _print("✓", "Tüm testler tamamlandı!", G)
            else:
                _print("⚠", f"Testler bitti (exit code: {rc})", Y)
        else:
            # ── GUI Mode (default) ───────────────────────────────
            launch_gui()

    except KeyboardInterrupt:
        _print("", "\nKullanıcı tarafından durduruldu.", Y)
    finally:
        # Stop background process if we started it
        if main_proc and main_proc.poll() is None:
            _print("ℹ", "Main llama-server kapatılıyor...", C)
            main_proc.terminate()
        if dllm_proc and dllm_proc.poll() is None:
            _print("ℹ", "DLLM llama-server kapatılıyor...", C)
            dllm_proc.terminate()


if __name__ == "__main__":
    main()
