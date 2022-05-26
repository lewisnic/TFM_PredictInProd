
# write some code to build your image

FROM python:3.8.6-buster

COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt
COPY api /api
COPY TaxiFareModel / TaxiFareModel

RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 -p 8000
