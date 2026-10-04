from pathlib import Path

from typer.testing import CliRunner

from ttr.cli import app

runner = CliRunner()


def test_synth_ingest_clean(tmp_path: Path) -> None:
    raw, interim = tmp_path / "raw", tmp_path / "interim"
    result = runner.invoke(app, ["synth", "--out", str(raw), "--n-pairs", "20"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app, ["ingest", "--raw-dir", str(raw), "--out-dir", str(interim), "--synthetic"]
    )
    assert result.exit_code == 0, result.output
    assert "n_cases: 20" in result.output
    assert (interim / "daily.parquet").exists()
    processed = tmp_path / "processed"
    result = runner.invoke(
        app, ["clean", "--interim-dir", str(interim), "--out-dir", str(processed)]
    )
    assert result.exit_code == 0, result.output
    assert (processed / "players.parquet").exists()
