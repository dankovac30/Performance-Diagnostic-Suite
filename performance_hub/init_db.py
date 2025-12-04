import sqlite3
import os


db_path = os.path.join(os.path.dirname(__file__), 'performance_hub.db')
conn = sqlite3.connect(db_path)

cursor = conn.cursor()

cursor.execute("PRAGMA foreign_keys = ON;")

cursor.execute("""CREATE TABLE IF NOT EXISTS Athletes (
    athlete_id TEXT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    gender TEXT,
    discipline TEXT
)""")

cursor.execute("""CREATE TABLE IF NOT EXISTS Metrics_Data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    athlete_id TEXT,
    test_date TEXT,
    trial_id TEXT,
    source TEXT,
    exercise TEXT,
    metric_name TEXT,
    value REAL,

    FOREIGN KEY (athlete_id) REFERENCES Athletes(athlete_id) ON DELETE CASCADE,

    UNIQUE(athlete_id, trial_id, metric_name)
)""")

conn.commit()

conn.close()