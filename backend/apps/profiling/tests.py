import pandas as pd
from django.test import SimpleTestCase

from .services import build_dataset_profile_response


class ProfilingServiceTests(SimpleTestCase):
    def test_profile_response_has_stable_placeholder_shape(self):
        dataset = type("DatasetStub", (), {"id": 1})()
        df = pd.DataFrame({"name": ["A", None], "amount": [10, 10]})

        profile = build_dataset_profile_response(dataset, df)

        self.assertEqual(profile["dataset_id"], 1)
        self.assertEqual(profile["summary"]["row_count"], 2)
        self.assertEqual(profile["summary"]["column_count"], 2)
        self.assertIn("columns", profile)
        self.assertEqual(len(profile["columns"]), 2)
