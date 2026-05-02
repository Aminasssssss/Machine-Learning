import sqlite3
import pandas as pd
from sklearn.datasets import load_breast_cancer

DB_PATH = "predictions.db"


def setup_database():
    data = load_breast_cancer()
    feature_names = list(data.feature_names)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS input_data")
    cursor.execute("DROP TABLE IF EXISTS predictions")

    feature_cols_sql = ",\n    ".join([f'"{name}" REAL' for name in feature_names])
    cursor.execute(f"""
        CREATE TABLE input_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {feature_cols_sql}
        )
    """)

    cursor.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY,
            prediction INTEGER,
            diagnosis TEXT,
            probability_benign REAL,
            probability_malignant REAL,
            prediction_timestamp TEXT,
            FOREIGN KEY (id) REFERENCES input_data(id)
        )
    """)

    df = pd.DataFrame(data.data, columns=feature_names).head(50)
    cols = ", ".join([f'"{c}"' for c in feature_names])
    placeholders = ", ".join(["?" for _ in feature_names])

    for _, row in df.iterrows():
        cursor.execute(f'INSERT INTO input_data ({cols}) VALUES ({placeholders})', tuple(row.values))

    conn.commit()
    conn.close()

    print(f"Database created: {DB_PATH}")
    print(f"Inserted {len(df)} rows into input_data")


if __name__ == "__main__":
    setup_database()