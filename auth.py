import sqlite3
import os
import time

DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def init_db():
    """初始化 SQLite 資料庫與用戶資料表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            provider_id TEXT NOT NULL,
            username TEXT NOT NULL,
            email TEXT,
            avatar_url TEXT,
            elo_rating INTEGER DEFAULT 1200,
            games_played INTEGER DEFAULT 0,
            created_at REAL,
            UNIQUE(provider, provider_id)
        )
    ''')
    conn.commit()
    conn.close()

def get_or_create_social_user(provider, provider_id, username, email=None, avatar_url=None):
    """查詢或自動註冊第三方社群用戶 (Google / FB / IG)"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, provider, provider_id, username, email, avatar_url, elo_rating, games_played
        FROM users WHERE provider = ? AND provider_id = ?
    ''', (provider, provider_id))
    row = cursor.fetchone()

    if row:
        user = {
            'id': row[0],
            'provider': row[1],
            'provider_id': row[2],
            'username': row[3],
            'email': row[4],
            'avatar_url': row[5],
            'elo_rating': row[6],
            'games_played': row[7]
        }
    else:
        # 新增用戶，預設 Elo 為 1200
        now = time.time()
        cursor.execute('''
            INSERT INTO users (provider, provider_id, username, email, avatar_url, elo_rating, games_played, created_at)
            VALUES (?, ?, ?, ?, ?, 1200, 0, ?)
        ''', (provider, provider_id, username, email, avatar_url, now))
        conn.commit()
        new_id = cursor.lastrowid
        user = {
            'id': new_id,
            'provider': provider,
            'provider_id': provider_id,
            'username': username,
            'email': email,
            'avatar_url': avatar_url,
            'elo_rating': 1200,
            'games_played': 0
        }

    conn.close()
    return user

if __name__ == '__main__':
    user = get_or_create_social_user('google', 'test_123', 'Pete Demo', 'pete@example.com')
    print("測試用戶寫入/讀取成功:", user)
