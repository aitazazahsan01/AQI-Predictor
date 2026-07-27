"""Single place that knows how to log in to Hopsworks. Everything else
(feature groups, feature views, model registry) is reached through the
`project`/`feature_store` objects returned here.
"""

import os

import hopsworks


def get_project():
    return hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ["HOPSWORKS_PROJECT_NAME"],
    )


def get_feature_store():
    return get_project().get_feature_store()


def get_model_registry():
    return get_project().get_model_registry()
