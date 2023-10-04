from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def create_dburl():
  hostname = $DB_HOST
  username = $DB_USER
  password = $DB_PASSWORD
  port = $DB_PORT
  database = $DB_NAME
  return f"postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database}"

engine = create_engine(create_dburl())
Session = sessionmaker(bind=engine)