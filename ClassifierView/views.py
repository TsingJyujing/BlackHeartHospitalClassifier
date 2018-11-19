from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
import jieba
import json
from sklearn.externals import joblib
from scipy.sparse import coo_matrix
import numpy

from util.http_response import response_json

# load model from file
model = joblib.load("model/hospital-classifier-logistic.m")

# Load words map from file
with open("model/words_map.json") as fp:
    words_map = json.load(fp)


def predict(hospital_name: str) -> bool:
    """
    predict result by hospital name
    :param hospital_name:
    :return: True for normal hospital, False for Putian hospital
    """
    words_index = [words_map[w] for w in jieba.cut(hospital_name) if w in words_map]
    if len(words_index) == 0:
        raise Exception("Can't classifier cause words are all not found.")

    il = [0 for _ in words_index]
    vl = [1 for _ in words_index]
    result = model.predict(
        coo_matrix(
            (
                numpy.array(vl),
                (
                    numpy.array(il),
                    numpy.array(words_index)
                )
            ),
            shape=(1, len(words_map))
        )
    )
    return bool(result[0] > 0)


@response_json
def classifier_api(request: HttpRequest):
    """
    Classifier API
    :param request:
    :return:
    """
    hospital_name = request.GET["name"]
    return {
        "status": "success",
        "result": predict(hospital_name),
        "input": hospital_name
    }


def index_page(request: HttpRequest) -> HttpResponse:
    """
    Show index page
    :param request:
    :return:
    """
    return render(request, "classifier.html")
