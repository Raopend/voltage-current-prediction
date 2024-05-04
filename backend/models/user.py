from datetime import datetime

from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    username: str
    email: str
    phone: str
    alert_option: str
    token: str
    notes: str
    created_at: datetime | None = Field(default=datetime.now())


class UserBase(SQLModel):
    username: str
    email: str
    phone: str
    alert_option: str
    token: str
    notes: str


class UserCreate(UserBase):
    pass


class UserRead(UserBase):
    id: int


class UserUpdate(UserBase):
    pass
