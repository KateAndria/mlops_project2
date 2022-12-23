import pytest
import pandas as pd
import model
from model import BaseModel


@pytest.fixture()
def get_fixture_data(scope="module"):
    df = pd.read_csv('/Users/eandrianova/PycharmProjects/pythonProject5/data/heart.csv', sep=",")
    return df

def test_init(mocker, get_fixture_data):
    def copy():
        return get_fixture_data.copy()

    mocker.patch.object(model, "get_postgress_data", copy)
    model_logreg = BaseModel("logreg")
    model_svc = BaseModel("svc")


def test_duplicated_rows(mocker, get_fixture_data):
    def copy():
        return get_fixture_data.copy()

    def duplicated_rows_df(df):
        return df.duplicated().any()

    assert duplicated_rows_df(copy()), "Alert! Duplicates in dataset"


def test_data_columns(mocker, get_fixture_data):
    def copy():
        return get_fixture_data.copy()

    def get_data_columns(df):
        return set(df.columns)

    reference_columns = set(["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh",
                         "exng", "oldpeak", "slp", "caa", "thall", "output"])

    assert get_data_columns(copy()) == reference_columns, f"Alert! Lost {reference_columns - get_data_columns(copy())} column"


@pytest.mark.skip(reason="This data is not historical")
def test_historical_data(mocker, get_fixture_data):
    def copy():
        return get_fixture_data.copy()

    reference_historical_len = 303

    assert copy().shape[0] != reference_historical_len, 'Alert! There is no historical data'


def test_target_deviations(mocker, get_fixture_data):
    def copy():
        return get_fixture_data.copy()

    reference_mean = 0.5445544554455446
    reference_std = 0.4988347841643926

    borders = [reference_mean - 3 * reference_std, reference_mean + 3 * reference_std]

    assert (copy()['output'].mean() > borders[0]) & (copy()['output'].mean() < borders[1]),\
        "Alert! Big deviation for tagret"


@pytest.mark.parametrize("model_name", ["logreg", "svc"])
def test_fit(mocker, get_fixture_data, model_name):
    def copy():
        return get_fixture_data.copy()

    mocker.patch.object(model, "get_postgress_data", copy)

    base_model = BaseModel(model_name)
    base_model.fit()

    assert base_model.model_is_trained
    
