# noinspection PyBroadException
import json
import traceback
import sys
from django.http import HttpResponse, HttpRequest
from bson import json_util


def get_json_response(obj):
    """
    Create a http response
    :param obj: object which json serializable
    :return:
    """
    return HttpResponse(json_util.dumps(obj))


def get_host(request: HttpRequest) -> str:
    """
    Get host info from request META
    :param request:
    :return:
    """
    return request.META["HTTP_HOST"].split(":")[0]


def get_remote_addr(request: HttpRequest) -> str:
    try:
        if "HTTP_X_FORWARDED_FOR" in request.META:
            return request.META['HTTP_X_FORWARDED_FOR']
        else:
            return request.META['REMOTE_ADDR']
    except Exception as ex:
        pass


def response_json(func):
    """
    Trying to run function, if exception caught, return error details with json format, else return json formatted object
    :param func:
    :return:
    """

    def wrapper(request: HttpRequest) -> HttpResponse:
        try:
            data = func(request)
            return get_json_response(data)
        except Exception as ex:
            print(traceback.format_exc(), file=sys.stderr)
            return get_json_response({
                "status": "error",
                "error_info": str(ex),
                "trace_back": traceback.format_exc()
            })

    return wrapper


def error_response(module_name="Mod", error_message="Error while processing request."):
    """
    Return error-report in JSON & HTTP format.
    :param module_name: Which the error happened.
    :param error_message: Error message throw
    :return: 
    """
    return HttpResponse(json.dumps({
        "status": "error",
        "error_module": module_name,
        "error_message": error_message
    }))
