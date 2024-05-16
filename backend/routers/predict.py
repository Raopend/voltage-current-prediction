from datetime import datetime, timedelta

from datamodel import PredictedResult, TimeSeriesFeatures
from dateutil.relativedelta import relativedelta
from dependencies import get_db_session, get_onnx_session
from fastapi import APIRouter, Depends
from lacheck import Get_detect_data
from sqlalchemy import Column, DateTime, Double, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

Base = declarative_base()


class EquipmentState(Base):
    __tablename__ = "dnaq_1Min_interval"

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


def predict(
    data: TimeSeriesFeatures, model_path: str, scaler_max=1, scaler_min=0
) -> PredictedResult:
    session = get_onnx_session(model_path=model_path)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    predicted = session.run(
        output_names=[label_name],
        input_feed={
            input_name: data.to_numpy(scaler_max=scaler_max, scaler_min=scaler_min)
        },
    )
    return PredictedResult(
        **{
            "predicted": PredictedResult.transform(
                predicted[0][0][0], scaler_max=scaler_max, scaler_min=scaler_min
            )
        }
    )


@router.get("/predict_I")
def post_predict_I(db: Session = Depends(get_db_session)):
    # 查询最近1分钟内的数据
    current_time = datetime.now() - relativedelta(years=2)
    fifteen_min_ago = current_time - timedelta(minutes=12)
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
        (current_time + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    )
    return {
        "collection_time": collection_time,
        "Ia": Ia,
        "Ib": Ib,
        "Ic": Ic,
        "predicted": predict(
            TimeSeriesFeatures(sequence=Ia),
            model_path="notebook/model/lstm_12_step.onnx",
            # scaler_max, scaler_min 分别对应训练数据的最大最小值, Ia
            scaler_max=373,
            scaler_min=0,
        ).predicted,
    }


@router.get("/predict_U")
def post_predict_U(db: Session = Depends(get_db_session)):
    # 查询最近300分钟内的数据
    current_time = datetime.now() - relativedelta(years=2)
    fifteen_min_ago = current_time - timedelta(minutes=12)
    data = (
        db.query(EquipmentState)
        .filter(EquipmentState.collection_time >= fifteen_min_ago)
        .filter(EquipmentState.collection_time <= current_time)
        .all()
    )
    Ua = [d.Ua for d in data]
    Ub = [d.Ub for d in data]
    Uc = [d.Uc for d in data]
    collection_time = [d.collection_time for d in data]
    collection_time.append(
        (current_time + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    )
    return {
        "collection_time": collection_time,
        "Ua": Ua,
        "Ub": Ub,
        "Uc": Uc,
        "predicted": predict(
            TimeSeriesFeatures(sequence=Ua),
            model_path="notebook/model/lstm_12_step_Ua.onnx",
            # scaler_max, scaler_min 分别对应训练数据的最大最小值, Ua
            scaler_max=373,
            scaler_min=0,
        ).predicted,
    }


@router.get("/check_U")
def check_U():
    data_path = "/Users/raopend/Workspace/voltage-current-prediction/backend/data/dnaq_history_data_2022_ext2.csv"
    model_path = "/Users/raopend/Workspace/voltage-current-prediction/backend/data/kmeans_model.joblib"
    X_test_original, distances, mean, std = Get_detect_data(data_path, model_path)
    electricList = X_test_original.tolist()
    return {
        "electricList": electricList,
        "distances": distances,
        "mean": mean,
        "std": std,
    }
