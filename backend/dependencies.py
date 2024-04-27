from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class MySQLSession:
    MYSQL_URL = "mysql+pymysql://root:123456@localhost:3306/dnaq?charset=utf8"
    POOL_SIZE = 20
    POOL_RECYCLE = 3600
    POOL_TIMEOUT = 15
    MAX_OVERFLOW = 2
    CONNECT_TIMEOUT = 60

    @classmethod
    def get_db_connection(cls):
        engine = create_engine(
            cls.MYSQL_URL,
            pool_size=cls.POOL_SIZE,
            pool_recycle=cls.POOL_RECYCLE,
            pool_timeout=cls.POOL_TIMEOUT,
            max_overflow=cls.MAX_OVERFLOW,
        )
        return engine

    @classmethod
    def get_db_session(cls):
        try:
            engine = cls.get_db_connection()
            Session = sessionmaker(engine)
            session = Session()
            return session
        except Exception as e:
            print("Error getting DB session:", e)
            return None


def get_db_session():
    return MySQLSession.get_db_session()
