import onnxruntime as rt
from sqlmodel import Session, SQLModel, create_engine


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
            echo=True,
        )
        return engine

    @classmethod
    def get_db_session(cls):
        try:
            engine = cls.get_db_connection()
            SQLModel.metadata.create_all(engine)
            return Session(engine)
        except Exception as e:
            print("Error getting DB session:", e)
            return None


def get_db_session():
    return MySQLSession.get_db_session()


def get_onnx_session(model_path: str):
    session = rt.InferenceSession(model_path)
    return session
