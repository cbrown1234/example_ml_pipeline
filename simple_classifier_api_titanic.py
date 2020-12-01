from enum import IntEnum, Enum
from typing import Optional

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from joblib import load
import pandas as pd


class PClass(IntEnum):
    first = 1
    second = 2
    third = 3


class Embarked(str, Enum):
    S = 'S'
    C = 'C'
    Q = 'Q'


class Sex(str, Enum):
    male = 'male'
    female = 'female'


app = FastAPI()


def to_camel(string: str) -> str:
    return ''.join(word.capitalize() for word in string.split('_'))


class Passenger(BaseModel):
    passenger_id: int
    pclass: PClass
    name: str
    sex: Sex
    age: Optional[float] = None
    sib_sp: int
    parch: int
    ticket: str
    fare: float
    cabin: Optional[str] = None
    embarked: Optional[Embarked] = None

    class Config:
        alias_generator = to_camel


class PassengerResponse(Passenger):
    prediction: bool


clf = load('clf_2.joblib')


@app.post("/predict/", response_model=PassengerResponse)
async def predict(passenger: Passenger):
    df_instance = pd.DataFrame([jsonable_encoder(passenger)])
    prediction = clf.predict(df_instance).tolist()[0]
    response = passenger.dict(by_alias=True)
    response.update({'Prediction': prediction})
    return response
