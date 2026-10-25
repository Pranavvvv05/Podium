from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy import Column

from database import Base, engine


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    target_column = Column(String(255))
    problem_type = Column(String(30))         
    n_rows = Column(Integer)
    n_features = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    jobs = relationship("Job", back_populates="dataset", cascade="all, delete-orphan")
    logs = relationship("BossLog", back_populates="dataset", cascade="all, delete-orphan")


class Job(Base):
    """One job = one model ("employee") training on one dataset."""
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    status = Column(String(20), default="queued")   
    current_task = Column(String(255))             
    score = Column(Float)                        
    metrics = Column(JSON)                          
    best_params = Column(JSON)
    model_path = Column(String(500))                
    error = Column(Text)
    seconds = Column(Float)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)

    dataset = relationship("Dataset", back_populates="jobs")


class BossLog(Base):
    __tablename__ = "boss_log"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="logs")


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("Tables created:", list(Base.metadata.tables.keys()))