import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("DATABASE_URL")
engine = create_engine(url)

with engine.connect() as conn:
    resultado = conn.execute(text(
        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'usuarios'"
    ))
    for fila in resultado:
        print(fila.indexname)
        print("  ", fila.indexdef)
        print()
