"""Trusted-proxy mode: one key, several clients named by a header.

Unit cases read `routellm.auth` directly; the server cases are in
`test_auth_client_header_server.py`.
"""

import hashlib

import pytest

from routellm import auth
from routellm.config import ConfigError
from routellm.request_class import check_policy_principals

PROXY_KEY = "sk-proxy-0001"
PLAIN_KEY = "sk-plain-0002"
HDR = "X-Route-Client"


def sha(key):
    return hashlib.sha256(key.encode()).hexdigest()


def keys_file(tmp_path, proxy_extra=None):
    body = f"""
keys:
  - key_sha256: "{sha(PROXY_KEY)}"
    principal: proxy
    profile: base
    allowed_classes: [open]
{proxy_extra if proxy_extra is not None else DEFAULT_PROXY}
  - key_sha256: "{sha(PLAIN_KEY)}"
    principal: plain
    profile: plain-prof
    allowed_classes: [open]
"""
    p = tmp_path / "keys.yaml"
    p.write_text(body)
    return p


DEFAULT_PROXY = f"""    client_header: {HDR}
    clients:
      web:
        profile: web-prof
        allowed_classes: [open, internal]
      batch: {{}}
"""


@pytest.fixture
def resolver(tmp_path):
    return auth.load_keys_file(keys_file(tmp_path))


def test_listed_client_gets_a_derived_principal_with_its_own_settings(resolver):
    p = resolver.resolve(PROXY_KEY, {HDR: "web"})
    assert p == auth.Principal("proxy/web", "web-prof", ("open", "internal"))


def test_listed_client_falls_back_to_the_entry_profile_and_classes(resolver):
    assert resolver.resolve(PROXY_KEY, {HDR: "batch"}) == auth.Principal(
        "proxy/batch", "base", ("open",)
    )


def test_header_absent_gives_the_entrys_own_principal(resolver):
    assert resolver.resolve(PROXY_KEY, {}) == auth.Principal("proxy", "base", ("open",))
    assert resolver.resolve(PROXY_KEY) == auth.Principal("proxy", "base", ("open",))


def test_unlisted_client_is_refused_naming_the_header_not_the_value(resolver):
    long_value = "x" * 5000
    for value in ("ghost", long_value, "", "web "):
        with pytest.raises(auth.ClientNotListed) as ei:
            resolver.resolve(PROXY_KEY, {HDR: value})
        assert ei.value.header == HDR
        assert HDR in str(ei.value)
        assert value.strip() == "" or value not in str(ei.value)
        assert len(str(ei.value)) < 200


def test_header_lookup_is_case_insensitive_with_a_real_header_mapping(resolver):
    from starlette.datastructures import Headers

    h = Headers({"x-route-client": "web"})
    assert resolver.resolve(PROXY_KEY, h).principal == "proxy/web"


def test_entry_without_opt_in_ignores_the_header_entirely(resolver):
    for value in ("web", "proxy", "plain/web", "anything"):
        assert resolver.resolve(PLAIN_KEY, {HDR: value}) == auth.Principal(
            "plain", "plain-prof", ("open",)
        )


def test_unknown_key_stays_none_whatever_the_header(resolver):
    assert resolver.resolve("sk-nope", {HDR: "web"}) is None


def test_principals_lists_every_derived_principal_for_startup_checks(resolver):
    names = {p.principal for p in resolver.principals()}
    assert names == {"proxy", "proxy/web", "proxy/batch", "plain"}


def test_policy_principals_may_name_a_derived_principal(resolver):
    from routellm.policy import Policy, PolicyError

    ok = Policy.model_validate(
        {
            "classes": {"open": {"treatment": "raw", "reach": ["local"], "tiers": ["default"]}},
            "principals": {"proxy/web": {"classes": ["open"]}},
        }
    )
    check_policy_principals(ok, resolver)
    bad = Policy.model_validate(
        {
            "classes": {"open": {"treatment": "raw", "reach": ["local"], "tiers": ["default"]}},
            "principals": {"proxy/ghost": {"classes": ["open"]}},
        }
    )
    with pytest.raises(PolicyError):
        check_policy_principals(bad, resolver)


@pytest.mark.parametrize(
    "extra",
    [
        f"    client_header: {HDR}\n",  # header without clients
        f"    client_header: {HDR}\n    clients: {{}}\n",  # empty clients
        "    clients:\n      web: {}\n",  # clients without header
        "    client_header: 'Bad Header'\n    clients:\n      web: {}\n",
        "    client_header: 'X:Y'\n    clients:\n      web: {}\n",
        "    client_header: ''\n    clients:\n      web: {}\n",
        f"    client_header: {HDR}\n    clients:\n      '': {{}}\n",
        f"    client_header: {HDR}\n    clients:\n      'a b': {{}}\n",
        f"    client_header: {HDR}\n    clients:\n      web:\n        bogus: 1\n",
        f"    client_header: {HDR}\n    clients:\n      web:\n        key: sk-x\n",
        f"    client_header: {HDR}\n    clients:\n      web: {{}}\n    extra_field: 1\n",
    ],
)
def test_bad_trusted_proxy_entries_are_startup_errors(tmp_path, extra):
    with pytest.raises(ConfigError):
        auth.load_keys_file(keys_file(tmp_path, extra))


def test_derived_principal_colliding_with_another_entry_is_refused(tmp_path):
    body = f"""
keys:
  - key_sha256: "{sha(PROXY_KEY)}"
    principal: proxy
    client_header: {HDR}
    clients: {{web: {{}}}}
  - key_sha256: "{sha(PLAIN_KEY)}"
    principal: proxy/web
"""
    p = tmp_path / "k.yaml"
    p.write_text(body)
    with pytest.raises(ConfigError, match="proxy/web"):
        auth.load_keys_file(p)


def test_auth_section_cannot_carry_the_new_keys(tmp_path):
    for key in ("client_header", "clients"):
        with pytest.raises(ConfigError):
            auth.build_auth({"auth": {"keys_file": "k.yaml", key: "x"}}, None)


def test_auth_stays_a_pinned_system_section():
    from routellm.layer_pins import PINNED_KEYS

    assert "auth" in PINNED_KEYS
