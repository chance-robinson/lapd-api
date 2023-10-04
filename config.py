from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

def create_dburl():
  hostname = os.environ.get('DB_HOST')
  username = os.environ.get('DB_USER')
  password = os.environ.get('DB_PASSWORD')
  port = os.environ.get('DB_PORT')
  database = os.environ.get('DB_NAME')
  return f"postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database}"

engine = create_engine(create_dburl())
Session = sessionmaker(bind=engine)