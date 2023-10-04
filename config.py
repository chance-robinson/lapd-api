from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def create_dburl():
  hostname = 'localhost'
  username = 'postgres'
  password = 'postgres'
  port = '5432'
  database = 'lapd'
  return f"postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database}"

engine = create_engine(create_dburl())
Session = sessionmaker(bind=engine)