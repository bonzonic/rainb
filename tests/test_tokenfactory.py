import asyncio

import httpx

from rainb import config
from rainb.tokenfactory import TokenFactory, TokenFactoryError


def make(handler) -> TokenFactory:
    tf = TokenFactory(api_key="test")
    tf.http = httpx.AsyncClient(base_url="https://tf.test/v1", transport=httpx.MockTransport(handler))
    return tf


def reply(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_overloaded_primary_falls_back_to_nano():
    seen = []

    def handler(request):
        model = __import__("json").loads(request.content)["model"]
        seen.append(model)
        if model == config.CHAT_MODEL:
            return httpx.Response(529, json={"detail": "System is overloaded"})
        return reply("Uokatsu")

    out = asyncio.run(make(handler).chat([{"role": "user", "content": "x"}]))
    assert out == "Uokatsu"
    assert seen == [config.CHAT_MODEL, config.CHAT_FALLBACK_MODEL]


def test_timeout_falls_back():
    def handler(request):
        if __import__("json").loads(request.content)["model"] == config.CHAT_MODEL:
            raise httpx.ReadTimeout("slow")
        return reply("ok")

    assert asyncio.run(make(handler).chat([{"role": "user", "content": "x"}])) == "ok"


def test_bad_request_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"detail": "bad"})

    try:
        asyncio.run(make(handler).chat([{"role": "user", "content": "x"}]))
        raise AssertionError("expected an error")
    except TokenFactoryError as e:
        assert e.status == 400
    assert len(calls) == 1


def test_explicit_model_has_no_fallback():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(529, json={"detail": "overloaded"})

    try:
        asyncio.run(make(handler).chat([{"role": "user", "content": "x"}], model="some/model"))
    except TokenFactoryError:
        pass
    assert len(calls) == 1
