from typing import List

from dependencies import get_db_session
from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from models.user import User, UserCreate, UserRead

router = APIRouter()


@router.get("/users/", response_model=List[UserRead], tags=["user"])
def read_users(skip: int = 0, limit: int = 100, session=Depends(get_db_session)):
    users = session.query(User).offset(skip).limit(limit).all()
    return users


@router.post("/users/", response_model=User, tags=["user"])
def create_user(user: UserCreate, session=Depends(get_db_session)):
    db_user = User.model_validate(user)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


@router.put("/users/{user_id}", response_model=User, tags=["user"])
def update_user(user_id: int, user: UserCreate, session=Depends(get_db_session)):
    db_user = session.get(User, user_id)
    if not db_user:
        return HTTPException(status_code=404, detail="User not found")
    user_data = user.model_dump(exclude_unset=True)
    db_user.sqlmodel_update(user_data)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user
