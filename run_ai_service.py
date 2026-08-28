import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=== 框制 AlphaZero AI 服務系統啟動 ===")
    
    # 步驟 1: 執行語法與安全驗證
    python_exe = sys.executable
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    verify_script = os.path.join(base_dir, 'scratch', 'safe_verify.py')

    if os.path.exists(verify_script):
        print(f"[Safety Check] 執行安全語法校驗: {verify_script}")
        res = subprocess.run([python_exe, verify_script])
        if res.returncode != 0:
            print("[CRITICAL ERROR] 語法校驗失敗，拒絕啟動 AI 服務！")
            sys.exit(1)

    print("[1/2] 啟動後台 Self-Play & Continuous Trainer 學習守護進程...")
    daemon_script = os.path.join(os.path.dirname(__file__), 'pipeline_daemon.py')
    daemon_proc = subprocess.Popen([python_exe, daemon_script])

    print("[2/2] 啟動 API FastAPI 雲端推論伺服器 (Port 8000)...")
    app_script = os.path.join(os.path.dirname(__file__), 'app.py')
    app_proc = subprocess.Popen([python_exe, app_script])

    print("=== AI 服務全體服務運作中 (Ctrl+C 可安全關閉) ===")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Shutdown] 正在優雅關閉所有 AI 後台進程...")
        app_proc.terminate()
        daemon_proc.terminate()
        print("[Shutdown] 關閉完成。")

if __name__ == '__main__':
    main()
