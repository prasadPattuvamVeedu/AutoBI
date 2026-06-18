from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from apps.datasets.models import Dataset, DatasetVersion
from django.core.files.uploadedfile import SimpleUploadedFile


class DatasetCleaningApiTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(
            username="cleaning_user",
            email="cleaning@example.com",
            password="StrongPass123!",
        )

        self.dataset = Dataset.objects.create(
            owner=self.user,
            name="Cleaning Dataset",
            description="Dataset for cleaning tests",
            file_type="csv",
            file_size=1024,
            row_count=4,
            column_count=3,
        )

        self.dataset.file.save(
            "cleaning.csv",
            SimpleUploadedFile("cleaning.csv", b"id,name,value\n1,A,10\n1,A,10\n2,B,\n3,C,100\n"),
        )
        self.dataset.save()

    def test_cleaning_report_requires_authentication(self):
        url = reverse("dataset-cleaning-report", kwargs={"id": self.dataset.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_cleaning_report_returns_suggestions(self):
        self.client.force_authenticate(self.user)
        url = reverse("dataset-cleaning-report", kwargs={"id": self.dataset.id})

        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("summary", response.data)
        self.assertIn("recommendations", response.data)
        self.assertGreaterEqual(len(response.data["recommendations"]), 1)

    def test_cleaning_report_uses_latest_dataset_version(self):
        self.client.force_authenticate(self.user)

        # Create a newer version file with no missing values and no duplicates.
        DatasetVersion.objects.create(
            dataset=self.dataset,
            version_number=1,
            file=SimpleUploadedFile(
                "cleaned.csv",
                b"id,name,value\n1,A,10\n2,B,20\n3,C,30\n",
                content_type="text/csv",
            ),
            is_cleaned=True,
            transformation_log={"action": "manual_clean"},
        )

        url = reverse("dataset-cleaning-report", kwargs={"id": self.dataset.id})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["summary"]["missing_values"], 0)
        self.assertEqual(response.data["duplicates"]["duplicate_count"], 0)
        self.assertEqual(response.data["summary"]["outliers"], 0)

    def test_apply_safe_fixes_creates_new_version(self):
        self.client.force_authenticate(self.user)
        url = reverse("dataset-cleaning-apply", kwargs={"id": self.dataset.id})

        response = self.client.post(url, {"apply_safe": True}, format="json")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(DatasetVersion.objects.filter(dataset=self.dataset).count(), 1)
        self.assertIsNotNone(response.data.get("transformation_log"))

    def test_versions_endpoint_returns_list(self):
        self.client.force_authenticate(self.user)
        url = reverse("dataset-versions", kwargs={"id": self.dataset.id})

        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("versions", response.data)

    def test_rollback_creates_new_version_from_existing(self):
        self.client.force_authenticate(self.user)
        apply_url = reverse("dataset-cleaning-apply", kwargs={"id": self.dataset.id})
        self.client.post(apply_url, {"apply_safe": True}, format="json")
        version = DatasetVersion.objects.filter(dataset=self.dataset).first()

        rollback_url = reverse("dataset-rollback", kwargs={"id": self.dataset.id})
        response = self.client.post(rollback_url, {"version_id": version.id}, format="json")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(DatasetVersion.objects.filter(dataset=self.dataset).count(), 2)
        self.assertEqual(response.data["transformation_log"]["action"], "rollback")
