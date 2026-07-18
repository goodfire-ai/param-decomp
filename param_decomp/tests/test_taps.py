"""The tap-address grammar: round-trips, and the wire keys targets/eval already use."""

from param_decomp.site_tree import ResidIn, SiteInput, parse_tap, tap_key


def test_tap_key_round_trips():
    for tap in (
        ResidIn(0),
        ResidIn(31),
        SiteInput("h.3.attn.q_proj"),
        SiteInput("layers.18.mlp.down_proj"),
    ):
        assert parse_tap(tap_key(tap)) == tap


def test_wire_forms_are_the_historical_keys():
    assert tap_key(ResidIn(18)) == "resid.18"
    assert tap_key(SiteInput("h.0.mlp.c_fc")) == "h.0.mlp.c_fc"
    assert parse_tap("resid.7") == ResidIn(7)
    assert parse_tap("h.2.attn.v_proj") == SiteInput("h.2.attn.v_proj")
