from datetime import datetime, timedelta

import onnxruntime as rt
from config import MODEL_PATH
from datamodel import PredictedResult, TimeSeriesFeatures
from dateutil.relativedelta import relativedelta
from dependencies import get_db_session
from fastapi import APIRouter, Depends
from sqlalchemy import Column, DateTime, Double, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

Base = declarative_base()


class EquipmentState(Base):
    __tablename__ = "dnaq_15Min_interval"

    id = Column(Integer, primary_key=True)
    Ia = Column(Double)
    Ib = Column(Double)
    Ic = Column(Double)
    Ua = Column(Double)
    Ub = Column(Double)
    Uc = Column(Double)
    elec_degree = Column(Double)
    P = Column(Double)
    COS = Column(Double)
    collection_time = Column(DateTime)


router = APIRouter()
session = rt.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name


def predict(data: TimeSeriesFeatures) -> PredictedResult:
    predicted = session.run(
        output_names=[label_name], input_feed={input_name: data.to_numpy()}
    )
    return PredictedResult(
        **{"predicted": PredictedResult.transform(predicted[0][0][0])}
    )


@router.get("/predict")
def post_predict(db: Session = Depends(get_db_session)):
    # 查询最近300分钟内的数据
    current_time = datetime.now() - relativedelta(years=2)
    fifteen_min_ago = current_time - timedelta(minutes=180)
    data = (
        db.query(EquipmentState)
        .filter(EquipmentState.collection_time >= fifteen_min_ago)
        .filter(EquipmentState.collection_time <= current_time)
        .all()
    )
    Ia = [d.Ia for d in data]
    Ib = [d.Ib for d in data]
    Ic = [d.Ic for d in data]
    collection_time = [d.collection_time for d in data]
    collection_time.append(
        (current_time + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
    )
    return {
        "collection_time": collection_time,
        "Ia": Ia,
        "Ib": Ib,
        "Ic": Ic,
        "predicted": predict(TimeSeriesFeatures(sequence=Ia)).predicted,
    }
