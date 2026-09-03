"""The CLI can say everything the web form can. It could start only a plain
research run — no categories, no memory-off, no briefs, no claim checks."""
import pytest

from app.cli import _params_from_args, build_parser


def _parse(*argv):
    return build_parser().parse_args(["run", *argv])


def test_defaults_match_the_web_form():
    p = _params_from_args(_parse("solid state batteries"))
    assert (p.depth, p.recency, p.kind) == (3, "all", "research")
    assert p.use_prior is True and p.categories == "" and p.brief_id is None
    assert p.origin == "cli" and p.created_by == "CLI"


def test_every_new_option_reaches_run_params():
    p = _params_from_args(_parse("Brief: Local LLM", "--kind", "brief",
                                 "--brief-id", "7", "--categories", " it,q&a ",
                                 "--no-prior", "-d", "6", "-r", "week"))
    assert p.kind == "brief" and p.brief_id == 7
    assert p.categories == "it,q&a"
    assert p.use_prior is False
    assert (p.depth, p.recency) == (6, "week")


def test_a_claim_check_reads_its_document_from_a_file(tmp_path):
    doc = tmp_path / "claims.md"
    doc.write_text("MLX doubles decode throughput on Apple Silicon.")
    p = _params_from_args(_parse("Claim check", "--kind", "verify",
                                 "--document", str(doc)))
    assert p.kind == "verify"
    assert "doubles decode" in p.document


def test_a_claim_check_without_a_document_is_refused_plainly():
    with pytest.raises(SystemExit) as exc:
        _params_from_args(_parse("Claim check", "--kind", "verify"))
    assert "--document" in str(exc.value)


def test_bad_kind_and_bad_depth_are_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        _parse("q", "--kind", "essay")
    with pytest.raises(SystemExit):
        _parse("q", "--depth", "11")


def test_the_ab_command_parses_its_arms():
    from app.cli import build_parser, _parse_env
    args = build_parser().parse_args(["ab", "fly rod reel seat", "--depth", "3", "--env-a", "GAP_VARIANT=default",
                                      "--env-b", "GAP_VARIANT=anchored, QUERY_SCOPES=off"])
    assert args.fn.__name__ == "_cmd_ab" and args.depth == 3
    assert _parse_env(args.env_b) == {"GAP_VARIANT": "anchored", "QUERY_SCOPES": "off"}
