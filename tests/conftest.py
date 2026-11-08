import os
import shutil
import sys

import pytest

# Spark workers must use the same interpreter as the driver.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


@pytest.fixture(scope="session")
def spark():
    pytest.importorskip("pyspark")
    if not (shutil.which("java") or os.environ.get("JAVA_HOME")):
        pytest.skip("Java is required for local Spark tests")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("tax-rag-tests")
        .config("spark.sql.ansi.enabled", "true")  # serverless default
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    # Same mechanism production uses: workers don't have `tax_rag` installed here either.
    from tax_rag.spark_support import ship_package_by_value

    ship_package_by_value()
    yield session
    session.stop()


@pytest.fixture
def mock_api():
    from tests.mock_tax_api import MockState, MockTaxApi, load_documents

    with MockTaxApi(MockState(load_documents(), token="test-token")) as api:
        yield api
