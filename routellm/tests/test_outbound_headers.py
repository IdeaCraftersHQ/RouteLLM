"""Headers of one outbound upstream call: merged, with the hop guard last."""

from routellm.controller import HOP_HEADER, outbound_headers


def test_the_hop_header_is_always_present():
    assert outbound_headers() == {HOP_HEADER: "1"}
    assert outbound_headers(None, {}, None) == {HOP_HEADER: "1"}


def test_sources_merge_and_later_ones_win():
    merged = outbound_headers({"X-A": "1", "X-B": "endpoint"}, {"X-B": "request"}, {"X-C": "pay"})

    assert merged == {"X-A": "1", "X-B": "request", "X-C": "pay", HOP_HEADER: "1"}


def test_no_source_can_switch_the_guard_off_under_another_spelling():
    merged = outbound_headers({"x-routellm-hop": ""}, {"X-ROUTELLM-HOP": "0"})

    hops = [k for k in merged if k.lower() == HOP_HEADER.lower()]
    assert hops == [HOP_HEADER]
    assert merged[HOP_HEADER] == "1"


def test_inputs_are_not_mutated():
    endpoint = {"X-A": "1"}

    outbound_headers(endpoint)

    assert endpoint == {"X-A": "1"}
