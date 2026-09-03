"""
База данных учёта персонала и журнала входов/выходов (SQLite).

Логика отметок:
  - Каждое распознавание чередует статус: вход -> выход -> вход -> ...
  - Защита от повторов (cooldown): если человека только что отметили,
    в течение COOLDOWN_SECONDS повторные распознавания игнорируются.
"""
import os
import sqlite3
import datetime
from contextlib import closing

COOLDOWN_SECONDS = 5          # пауза между отметками одного человека
# Кладём БД рядом со скриптом, а не в текущую рабочую папку.
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")


class AttendanceDB:
    def __init__(self, db_file: str = DB_FILE, cooldown: int = COOLDOWN_SECONDS):
        self.db_file = db_file
        self.cooldown = cooldown
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with closing(self._connect()) as conn, conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS staff (
                    person_id   TEXT PRIMARY KEY,
                    full_name   TEXT NOT NULL,
                    position    TEXT,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id   TEXT NOT NULL,
                    full_name   TEXT,
                    direction   TEXT NOT NULL,      -- 'IN' или 'OUT'
                    timestamp   TEXT NOT NULL,
                    confidence  REAL
                )
            """)

    # ─── Персонал ──────────────────────────────────────────────
    def add_staff(self, person_id: str, full_name: str, position: str = ""):
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO staff (person_id, full_name, position, created_at) "
                "VALUES (?, ?, ?, ?)",
                (person_id, full_name, position, datetime.datetime.now().isoformat(timespec="seconds"))
            )

    def remove_staff(self, person_id: str):
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM staff WHERE person_id = ?", (person_id,))

    def get_staff(self):
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT person_id, full_name, position, created_at FROM staff ORDER BY full_name"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_staff_name(self, person_id: str) -> str:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT full_name FROM staff WHERE person_id = ?", (person_id,)
            ).fetchone()
        return row["full_name"] if row else person_id

    # ─── Журнал ────────────────────────────────────────────────
    def _last_event(self, person_id: str):
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT direction, timestamp FROM attendance "
                "WHERE person_id = ? ORDER BY id DESC LIMIT 1",
                (person_id,)
            ).fetchone()
        return row

    def register_scan(self, person_id: str, confidence: float = 0.0):
        """
        Регистрирует распознавание человека.
        Возвращает кортеж (status, direction):
          status: 'logged'   — отметка записана
                  'cooldown' — слишком рано, проигнорировано
          direction: 'IN' / 'OUT' (для записанной отметки) или None
        """
        now = datetime.datetime.now()
        last = self._last_event(person_id)

        if last is not None:
            last_time = datetime.datetime.fromisoformat(last["timestamp"])
            if (now - last_time).total_seconds() < self.cooldown:
                return "cooldown", last["direction"]
            # чередуем направление
            direction = "OUT" if last["direction"] == "IN" else "IN"
        else:
            direction = "IN"

        full_name = self.get_staff_name(person_id)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO attendance (person_id, full_name, direction, timestamp, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                (person_id, full_name, direction, now.isoformat(timespec="seconds"), confidence)
            )
        return "logged", direction

    def register_directed(self, person_id: str, direction: str, confidence: float = 0.0):
        """
        Регистрирует событие с ЗАДАННЫМ направлением (для схемы двух камер:
        камера входа всегда даёт 'IN', камера выхода — 'OUT').

        Логика игнорирования дублей:
          - повторный 'IN', когда человек уже внутри  -> 'duplicate'
          - повторный 'OUT', когда человек уже снаружи -> 'duplicate'
          - событие в течение cooldown после прошлого  -> 'cooldown'

        Возвращает (status, direction):
          status: 'logged' | 'cooldown' | 'duplicate'
        """
        assert direction in ("IN", "OUT")
        now = datetime.datetime.now()
        last = self._last_event(person_id)

        if last is not None:
            last_time = datetime.datetime.fromisoformat(last["timestamp"])
            if (now - last_time).total_seconds() < self.cooldown:
                return "cooldown", direction
            # человек уже в нужном состоянии — повтор игнорируем
            if last["direction"] == direction:
                return "duplicate", direction
        else:
            # первое в истории событие имеет смысл только как вход
            if direction == "OUT":
                return "duplicate", direction

        full_name = self.get_staff_name(person_id)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO attendance (person_id, full_name, direction, timestamp, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                (person_id, full_name, direction, now.isoformat(timespec="seconds"), confidence)
            )
        return "logged", direction

    def get_log(self, limit: int = 500, date_filter: str = None):
        """Возвращает журнал. date_filter='YYYY-MM-DD' фильтрует по дню."""
        query = ("SELECT id, person_id, full_name, direction, timestamp, confidence "
                 "FROM attendance")
        params = []
        if date_filter:
            query += " WHERE timestamp LIKE ?"
            params.append(date_filter + "%")
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def who_is_inside(self):
        """Список person_id, у которых последняя отметка = IN."""
        inside = []
        for s in self.get_staff():
            last = self._last_event(s["person_id"])
            if last and last["direction"] == "IN":
                inside.append(s["person_id"])
        return inside

    def export_log_csv(self, path: str, date_filter: str = None):
        import csv
        rows = self.get_log(limit=10_000_000, date_filter=date_filter)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["ID", "person_id", "ФИО", "Направление", "Время", "Сходство"])
            for r in reversed(rows):
                direction = "Вход" if r["direction"] == "IN" else "Выход"
                writer.writerow([r["id"], r["person_id"], r["full_name"],
                                 direction, r["timestamp"],
                                 f"{(r['confidence'] or 0) * 100:.1f}%"])
