import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=== 框制 AlphaZero AI 服務系統啟動 (Render 記憶體優化模式) ===")
    
    python_exe = sys.executable
    port = os.environ.get("PORT", "10000")
    print(f"[Server Port] 綁定通訊埠: {port}")

    print("[1/2] 啟動後台 Self-Play & Continuous Trainer 學習守護進程...")
    daemon_script = os.path.join(os.path.dirname(__file__), 'pipeline_daemon.py')
    daemon_proc = subprocess.Popen([python_exe, daemon_script])

    print(f"[2/2] 啟動 API FastAPI 雲端推論伺服器 (Port {port})...")
    app_script = os.path.join(os.path.dirname(__file__), 'app.py')
    env = os.environ.copy()
    env["PORT"] = str(port)
    app_proc = subprocess.Popen([python_exe, app_script], env=env)

    print("=== AI 全體服務運作中 ===")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Shutdown] 正在優雅關閉所有 AI 後台進程...")
        app_proc.terminate()
        daemon_proc.terminate()

if __name__ == '__main__':
    main()
