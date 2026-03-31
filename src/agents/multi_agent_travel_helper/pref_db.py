# -*- coding: utf-8 -*-
"""
旅行价位偏好 — SQLite 持久化

数据库文件位于包目录下 ``pref.db``（通常由 .gitignore 忽略）。仅存储与用户预算检索强相关的
三类偏好文案：交通、门票、酒店。创建表与首次连接采用懒初始化（``init_pref_db``）。

与主图的关系
------------
- ``hydrate_price_preferences``：会话开始若 state 中对应键为空，则用历史值预填。
- ``persist_price_preferences``：用户 intake 确认后写回，实现跨会话记忆。
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

# 与 multi_agent_travel_helper 包同级目录，便于与代码一同部署
_PKG_DIR = Path(__file__).resolve().parent
PREF_DB_PATH = _PKG_DIR / "pref.db"

_DDL = """
CREATE TABLE IF NOT EXISTS user_price_preferences (
    user_id TEXT PRIMARY KEY NOT NULL,
    travel_price_preference TEXT,
    site_price_preference TEXT,
    hotel_price_preference TEXT,
    updated_at INTEGER NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    """建立到 pref.db 的连接（短连接用法，由 with 管理关闭）。"""
    return sqlite3.connect(str(PREF_DB_PATH))


def init_pref_db() -> None:
    """若表不存在则创建；幂等，可反复调用。"""
    with _connect() as conn:
        conn.execute(_DDL)


def get_price_preferences(user_id: str) -> dict[str, str | None] | None:
    """按 user_id 读取一行偏好；无记录返回 None；空 user_id 会规范为 default_user。"""
    init_pref_db()
    uid = user_id or "default_user"
    with _connect() as conn:
        row = conn.execute(
            "SELECT travel_price_preference, site_price_preference, hotel_price_preference "
            "FROM user_price_preferences WHERE user_id = ?",
            (uid,),
        ).fetchone()
    if not row:
        return None
    return {
        "travel_price_preference": row[0],
        "site_price_preference": row[1],
        "hotel_price_preference": row[2],
    }


def upsert_price_preferences(
    user_id: str,
    *,
    travel_price_preference: str | None,
    site_price_preference: str | None,
    hotel_price_preference: str | None,
) -> None:
    """插入或整行覆盖更新三类偏好，并刷新 updated_at 为当前 Unix 时间戳。"""
    init_pref_db()
    uid = user_id or "default_user"
    now = int(time.time())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_price_preferences (
                user_id, travel_price_preference, site_price_preference,
                hotel_price_preference, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                travel_price_preference = excluded.travel_price_preference,
                site_price_preference = excluded.site_price_preference,
                hotel_price_preference = excluded.hotel_price_preference,
                updated_at = excluded.updated_at
            """,
            (uid, travel_price_preference, site_price_preference, hotel_price_preference, now),
        )


def delete_price_preferences(user_id: str) -> None:
    """按 user_id 删除偏好行；用于测试或用户重置（当前主图未调用）。"""
    init_pref_db()
    uid = user_id or "default_user"
    with _connect() as conn:
        conn.execute("DELETE FROM user_price_preferences WHERE user_id = ?", (uid,))
