import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

# Falls back to a local SQLite database if DATABASE_URL is not set on Render
DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///./podium.db"

# Fix for Render PostgreSQL URLs if you attach a Render Postgres database later
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

if __name__ == "__main__":
    with engine.connect() as conn:
        if DATABASE_URL.startswith("sqlite"):
            print("Connected to SQLite:", DATABASE_URL)
        else:
            print("Connected to:", conn.exec_driver_sql("select current_database()").scalar())