"""Single place that knows how to log in to Hopsworks. Everything else
(feature groups, feature views, model registry) is reached through the
`project`/`feature_store` objects returned here.
"""

import os

import hopsworks

REQUIRED_ENV_VARS = ("HOPSWORKS_API_KEY", "HOPSWORKS_PROJECT_NAME")


def _require_env() -> dict[str, str]:
    """Fails with an actionable message rather than a bare KeyError.

    In CI a missing secret otherwise surfaces as `KeyError: 'HOPSWORKS_API_KEY'`
    two seconds into the run, which says nothing about *why* it is missing.
    """
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ".\nLocally: add them to .env (see .env.example)."
            + "\nIn GitHub Actions: add them as *repository* secrets under"
            + " Settings > Secrets and variables > Actions, and check the names"
            + " match exactly (they are case-sensitive)."
        )
    return {name: os.environ[name] for name in REQUIRED_ENV_VARS}


def get_project():
    env = _require_env()
    return hopsworks.login(
        api_key_value=env["HOPSWORKS_API_KEY"],
        project=env["HOPSWORKS_PROJECT_NAME"],
    )


def get_feature_store():
    return get_project().get_feature_store()


def get_model_registry():
    return get_project().get_model_registry()
