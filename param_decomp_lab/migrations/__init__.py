"""One-shot config-schema migrations, kept in-repo so they land (and run) with the
schema change that needs them — never as a PR-comment attachment. Each module is a
`python -m` CLI over yaml files plus a pure raw-dict transform reusable on stored run
configs (`launch_config.yaml`s under runs/)."""
