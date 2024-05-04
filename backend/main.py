import json
from contextlib import asynccontextmanager

import requests
import sqlmodel
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from datamodel import TimeSeriesFeatures
from dependencies import get_db_session
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.user import User
from routers import predict, state, user
from routers.predict import EquipmentState
from sqlmodel import select


def send_alert(title: str, content: str, token: str):
    url = "http://www.pushplus.plus/send"
    data = {"token": token, "title": title, "content": content}
    body = json.dumps(data).encode(encoding="utf-8")
    headers = {"Content-Type": "application/json"}
    requests.post(url, data=body, headers=headers)


def send_batch_alert():
    # 获取所有用户
    session: sqlmodel.Session = get_db_session()
    users = session.exec(select(User)).all()
    # 获取设备参数
    data = session.exec(select(EquipmentState).limit(12)).all()
    Ua_sequence = [d.Ua for d in data]
    Ub_sequence = [d.Ub for d in data]
    Uc_sequence = [d.Uc for d in data]
    Ia_sequence = [d.Ia for d in data]
    Ib_sequence = [d.Ib for d in data]
    Ic_sequence = [d.Ic for d in data]
    average_Ua = sum([d.Ua for d in data]) / len(data)
    average_Ub = sum([d.Ub for d in data]) / len(data)
    average_Uc = sum([d.Uc for d in data]) / len(data)
    average_Ia = sum([d.Ia for d in data]) / len(data)
    average_Ib = sum([d.Ib for d in data]) / len(data)
    average_Ic = sum([d.Ic for d in data]) / len(data)
    for u in users:
        # 获取用户 options
        alert_options = str(u.alert_option).split(",")
        if "Ua" in alert_options:
            if (
                predict.predict(TimeSeriesFeatures(sequence=Ua_sequence)).predicted
                > average_Ua
            ):
                send_alert(
                    title="Ua 值异常", content=f"Ua 值为 {data.Ua}", token=u.token
                )
        elif "Ub" in alert_options:
            if (
                predict.predict(TimeSeriesFeatures(sequence=Ub_sequence)).predicted
                > average_Ub
            ):
                send_alert(
                    title="Ub 值异常", content=f"Ub 值为 {data.Ub}", token=u.token
                )
        elif "Uc" in alert_options:
            if (
                predict.predict(TimeSeriesFeatures(sequence=Uc_sequence)).predicted
                > average_Uc
            ):
                send_alert(
                    title="Uc 值异常", content=f"Uc 值为 {data.Uc}", token=u.token
                )
        elif "Ia" in alert_options:
            if (
                predict.predict(TimeSeriesFeatures(sequence=Ia_sequence)).predicted
                > average_Ia
            ):
                send_alert(
                    title="Ia 值异常", content=f"Ia 值为 {data.Ia}", token=u.token
                )
        elif "Ib" in alert_options:
            if (
                predict.predict(TimeSeriesFeatures(sequence=Ib_sequence)).predicted
                > average_Ib
            ):
                send_alert(
                    title="Ib 值异常", content=f"Ib 值为 {data.Ib}", token=u.token
                )
        elif "Ic" in alert_options:
            if (
                predict.predict(TimeSeriesFeatures(sequence=Ic_sequence)).predicted
                > average_Ic
            ):
                send_alert(
                    title="Ic 值异常", content=f"Ic 值为 {data.Ic}", token=u.token
                )
        else:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = BackgroundScheduler()
    scheduler.add_job(send_batch_alert, "interval", seconds=10)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(state.router)
app.include_router(user.router)
app.include_router(predict.router)


if __name__ == "__main__":
    uvicorn.run(
        app="main:app", host="0.0.0.0", port=8000, log_level="info", reload=True
    )
