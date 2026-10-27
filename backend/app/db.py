import time

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(retries: int = 20, delay: float = 1.5):
    from . import models  # noqa: F401

    for attempt in range(retries):
        try:
            Base.metadata.create_all(engine)
            return
        except OperationalError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
