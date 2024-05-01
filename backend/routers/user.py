from typing import List

import sqlmodel
from dependencies import get_db_session
from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import HTTPException
from models.user import User, UserCreate, UserRead
from sqlmodel import select

router = APIRouter()


@router.get("/users/", response_model=List[UserRead], tags=["user"])
def read_users(
    offset: int = 0,
    limit: int = Query(default=10),
    session: sqlmodel.Session = Depends(get_db_session),
):
    with session as s:
        users = s.exec(select(User).offset(offset).limit(limit)).all()
        return users


@router.post("/users/", response_model=User, tags=["user"])
def create_user(user: UserCreate, session: sqlmodel.Session = Depends(get_db_session)):
    with session as s:
        db_user = User.model_validate(user)
        s.add(db_user)
        s.commit()
        s.refresh(db_user)
        return db_user


@router.put("/users/{user_id}", response_model=User, tags=["user"])
def update_user(
    user_id: int, user: UserCreate, session: sqlmodel.Session = Depends(get_db_session)
):
    with session as s:
        db_user = s.get(User, user_id)
        if not db_user:
            return HTTPException(status_code=404, detail="User not found")
        user_data = user.model_dump(exclude_unset=True)
        db_user.sqlmodel_update(user_data)
        s.add(db_user)
        s.commit()
        s.refresh(db_user)
        return db_user


@router.delete("/users/{user_id}", tags=["user"])
def delete_user(user_id: int, session: sqlmodel.Session = Depends(get_db_session)):
    with session as s:
        db_user = s.get(User, user_id)
        if not db_user:
            return HTTPException(status_code=404, detail="User not found")
        s.delete(db_user)
        s.commit()
        return db_user
