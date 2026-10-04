# time-to-risk-it

**Player-risk early warning for online betting.** Survival models (Cox, XGBoost) predict *how soon*
an active player is likely to trigger a responsible-gambling (RG) intervention, and a decision layer
turns those predictions into an intervention policy: who to contact, at what threshold, and under
what weekly capacity.

> 🚧 Work in progress. Data pipeline and synthetic data are in place; modelling is next.

## Data

The project uses the bwin dataset *Behavioral Characteristics of Internet Gamblers Who Trigger
Corporate Responsible Gambling Interventions* (Gray, LaPlante & Shaffer, 2012), published by
[The Transparency Project](http://www.thetransparencyproject.org/) (Division on Addiction,
Cambridge Health Alliance, a Harvard Medical School teaching affiliate).

The data is licensed and **not redistributed here**. See [docs/data.md](docs/data.md) for how to
obtain it. Everything also runs on a schema-faithful **synthetic dataset**, so no registration is
needed to try the pipeline.

## Quickstart

```bash
uv sync
uv run ttr synth --out data/raw/synthetic   # generate synthetic data
uv run ttr ingest --raw-dir data/raw/synthetic --synthetic
uv run pytest
```

## License

Code: MIT. Data: subject to The Transparency Project terms.
