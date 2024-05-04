import asyncio
from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta
from dependencies import get_db_session
from fastapi import APIRouter, Depends
from models.equipment import EquipmentState, TimeInterval
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session
from starlette.websockets import WebSocket

router = APIRouter()


class TrendQueryParams(BaseModel):
    start_time: datetime
    end_time: datetime
    interval: TimeInterval


def get_date_format(interval: TimeInterval) -> str:
    if interval == TimeInterval.minute:
        return "%Y-%m-%d %H:%i:00"
    elif interval == TimeInterval.half_hour:
        # Rounds down to the nearest half-hour
        return (
            "%Y-%m-%d %H:30:00"
            if "30" <= datetime.strftime(datetime.now(), "%M") < "60"
            else "%Y-%m-%d %H:00:00"
        )
    elif interval == TimeInterval.hour:
        return "%Y-%m-%d %H:00:00"
    elif interval == TimeInterval.day:
        return "%Y-%m-%d"


@router.websocket("/ws/equipment_state")
async def get_equipment_state(
    websocket: WebSocket, session: Session = Depends(get_db_session)
):
    await websocket.accept()
    while True:
        _ = await websocket.receive_text()
        current_time = datetime.now() - relativedelta(years=2)
        fifteen_sec_ago = current_time - timedelta(seconds=15)
        data = (
            session.query(EquipmentState)
            .filter(EquipmentState.collection_time >= fifteen_sec_ago)
            .filter(EquipmentState.collection_time <= current_time)
            .limit(1)
            .all()
        )
        #  取出第一个，如果没有数据则返回空字典
        data = data[0].__dict__ if data else {}
        if data:
            data.pop("_sa_instance_state", None)
        await websocket.send_json({"data": data})
        await asyncio.sleep(10)


@router.post("/query/equipment_I_trend", tags=["equipment"])
async def get_equipment_I_trend(
    params: TrendQueryParams, db: Session = Depends(get_db_session)
):
    # params start_time 减去 2 年
    params.start_time = params.start_time - relativedelta(years=2)
    # params end_time 减去 2 年
    params.end_time = params.end_time - relativedelta(years=2)
    time_format = get_date_format(params.interval)
    query = (
        db.query(
            func.date_format(EquipmentState.collection_time, time_format).label(
                "time_interval"
            ),
            func.avg(EquipmentState.Ia).label("average_Ia"),
            func.avg(EquipmentState.Ib).label("average_Ib"),
            func.avg(EquipmentState.Ic).label("average_Ic"),
        )
        .filter(
            EquipmentState.collection_time >= params.start_time,
            EquipmentState.collection_time <= params.end_time,
        )
        .group_by("time_interval")
        .order_by("time_interval")
    )
    results = query.all()
    # 将查询结果转换为字典列表, 使用 EquipmentState.__table__.columns.keys() 获取所有列名，不要 id
    results = [dict(zip(["collection_time", "Ia", "Ib", "Ic"], row)) for row in results]
    # 将数值保留两位小数
    for result in results:
        for key in ["Ia", "Ib", "Ic"]:
            result[key] = round(result[key], 2)
    return results


@router.post("/query/equipment_U_trend", tags=["equipment"])
async def get_equipment_U_trend(
    params: TrendQueryParams, db: Session = Depends(get_db_session)
):
    # params start_time 减去 2 年
    params.start_time = params.start_time - relativedelta(years=2)
    # params end_time 减去 2 年
    params.end_time = params.end_time - relativedelta(years=2)
    time_format = get_date_format(params.interval)
    query = (
        db.query(
            func.date_format(EquipmentState.collection_time, time_format).label(
                "time_interval"
            ),
            func.avg(EquipmentState.Ua).label("average_Ua"),
            func.avg(EquipmentState.Ub).label("average_Ub"),
            func.avg(EquipmentState.Uc).label("average_Uc"),
        )
        .filter(
            EquipmentState.collection_time >= params.start_time,
            EquipmentState.collection_time <= params.end_time,
        )
        .group_by("time_interval")
        .order_by("time_interval")
    )
    results = query.all()
    # 将查询结果转换为字典列表, 使用 EquipmentState.__table__.columns.keys() 获取所有列名，不要 id
    results = [dict(zip(["collection_time", "Ua", "Ub", "Uc"], row)) for row in results]
    # 将数值保留两位小数
    for result in results:
        for key in ["Ua", "Ub", "Uc"]:
            result[key] = round(result[key], 2)
    return results
