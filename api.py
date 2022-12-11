from io import BytesIO
from model import BaseModel
from sqlalchemy import create_engine
from flask import Flask, jsonify
from flask_restx import Resource, Api, reqparse, fields

import os
import pickle
import psycopg2
import pandas as pd


if os.environ.get('RUNTIME_DOCKER', False):
    POSTGRES_HOST = os.environ['POSTGRES_HOST']
    POSTGRES_DB = os.environ['POSTGRES_DB']
    POSTGRES_USER = os.environ['POSTGRES_USER']
    POSTGRES_PASSWORD = os.environ['POSTGRES_PASSWORD']
else:
    POSTGRES_HOST = "localhost"
    POSTGRES_DB = "hwdb"
    POSTGRES_USER = "hwdb_user"
    POSTGRES_PASSWORD = "hwdb_passwd"
POSTGRES_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"


app = Flask(__name__)
api = Api(app, title='ML models', description='Heart Attack Analysis & Prediction Dataset')
upload_parser = api.parser()
models = {}

upload_parser.add_argument('model_name', required=True, location='args')
upload_parser.add_argument('model_params', required=False, location='args')

delete = api.model(
    'Delete', {
        'model_id': fields.String(required=True, title='Model id')
    })

train = api.model(
    'Train', {
        'model_id': fields.String(required=True, title='Model id')
    })

predict_sample = api.model(
    'Predict sample', {
        'model_id': fields.String(required=True, title='Model id')
    }
)


@api.route('/model/add')
class Add(Resource):
    @api.expect(upload_parser)
    @api.doc(
        responses={
            200: "Success",
            500: "Select other model (logreg or svc)"
        })
    def post(self):

        args = upload_parser.parse_args()
        model_name = args['model_name']
        model_params = args['model_params']
        clf = BaseModel(model_name=model_name, model_params={})
        model_id = clf.model_id

        buffer = BytesIO()
        pickle.dump(clf, buffer)
        buffer.seek(0)

        pg_client = create_engine(POSTGRES_URL)
        pg_client.execute(
                    f"""
                    INSERT INTO public.models ("model_id", "model_name", "model_params", "model_weights")
                    VALUES (%s,%s,%s,%s);
                    """,
                    (model_id, model_name, model_params, psycopg2.Binary(buffer.read()))
        )
        pg_client.dispose()
        msg = f"Model {model_name} added, id: {model_id}"
        return msg


@api.route("/model/list")
class List(Resource):
    @api.doc(responses={200: "Success"})
    def get(self):
        pg_client = create_engine(POSTGRES_URL)
        models = pd.read_sql_query('SELECT "model_id", "model_name", "model_params", "model_is_trained" FROM public.models;', pg_client).set_index("model_id").to_dict(orient="index")
        if len(models) == 0:
            return 'No models added'
        else:
            return models


@api.route("/model/delete")
class Delete(Resource):
    @api.expect(delete)
    @api.doc(
        responses={
            200: "Success",
            500: "No such model"
        })
    def delete(self):
        model_id = api.payload["model_id"]

        pg_client = create_engine(POSTGRES_URL)
        models = pd.read_sql_query('SELECT "model_id" FROM public.models', pg_client)['model_id'].tolist()
        pg_client.dispose()

        if model_id in models:
            pg_client = create_engine(POSTGRES_URL)
            pg_client.execute(
                f"""
                DELETE FROM public.models WHERE "model_id" = '{model_id}';
                """
            )
            pg_client.dispose()
            msg = f'Model {model_id} deleted'
        else:
            msg = 'No such model'
        return msg


@api.route("/model/train")
class Train(Resource):
    @api.expect(train)
    @api.doc(
        responses={
            200: "Success",
            500: "No such model"
        })
    def post(self):
        model_id = api.payload["model_id"]

        pg_client = create_engine(POSTGRES_URL)
        models = pd.read_sql_query('SELECT "model_id" FROM public.models', pg_client)['model_id'].tolist()
        pg_client.dispose()

        if model_id in models:
            pg_client = create_engine(POSTGRES_URL)
            model_raw = pg_client.execute(f"""SELECT "model_weights" FROM public.models WHERE "model_id" = '{model_id}';""").fetchone()[0]
            pg_client.dispose()

            model = pickle.loads(model_raw)
            model.fit()

            buffer = BytesIO()
            pickle.dump(model, buffer)
            buffer.seek(0)

            pg_client = create_engine(POSTGRES_URL)
            pg_client.execute(
                f"""
                UPDATE public.models
                SET
                    "model_is_trained" = True,
                    "model_weights" = %s
                WHERE "model_id" = '{model_id}';
                """,
                (psycopg2.Binary(buffer.read()))
            )
            pg_client.dispose()

            msg = f'Model {model_id} trained'
        else:
            msg = 'No such model'

        return msg


@api.route("/model/predict")
class Predict(Resource):
    @api.expect(predict_sample)
    @api.doc(
        responses={
            200: "Success",
            500: "No such model"
        })
    def post(self):
        model_id = api.payload["model_id"]

        pg_client = create_engine(POSTGRES_URL)
        models = pd.read_sql_query('SELECT "model_id" FROM public.models', pg_client)['model_id'].tolist()
        pg_client.dispose()

        if model_id in models:
            pg_client = create_engine(POSTGRES_URL)
            model_raw = pg_client.execute(
                f"""SELECT "model_weights" FROM public.models WHERE "model_id" = '{model_id}';""").fetchone()[0]
            pg_client.dispose()
            model = pickle.loads(model_raw)

            return jsonify(model.predict().tolist())
        else:
            return 'No such model'


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
