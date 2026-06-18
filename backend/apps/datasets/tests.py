import os
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from tempfile import TemporaryDirectory

from .models import ColumnSchema, Dataset, DatasetVersion


class DatasetCrudApiTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user_a = User.objects.create_user(
            username="dataset_user_a",
            email="dataset_a@example.com",
            password="StrongPass123!",
        )
        self.user_b = User.objects.create_user(
            username="dataset_user_b",
            email="dataset_b@example.com",
            password="StrongPass123!",
        )
        self.dataset_a = Dataset.objects.create(
            owner=self.user_a,
            name="Sales Dataset",
            description="Owned by user A",
            file_type="csv",
            file_size=1024,
            row_count=10,
            column_count=3,
        )
        DatasetVersion.objects.create(dataset=self.dataset_a, version_number=1)
        ColumnSchema.objects.create(
            dataset=self.dataset_a,
            column_name="revenue",
            detected_type="number",
            missing_count=0,
            unique_count=10,
            role="measure",
        )
        self.dataset_b = Dataset.objects.create(
            owner=self.user_b,
            name="Marketing Dataset",
            file_type="xlsx",
        )

    def test_user_sees_only_owned_datasets(self):
        self.client.force_authenticate(self.user_a)

        response = self.client.get(reverse("dataset-list"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["id"], self.dataset_a.id)

    def test_user_can_retrieve_owned_dataset_detail(self):
        self.client.force_authenticate(self.user_a)

        response = self.client.get(
            reverse("dataset-detail", kwargs={"id": self.dataset_a.id})
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["name"], "Sales Dataset")
        self.assertEqual(len(response.data["versions"]), 1)
        self.assertEqual(len(response.data["columns"]), 1)

    def test_user_cannot_retrieve_another_users_dataset(self):
        self.client.force_authenticate(self.user_b)

        response = self.client.get(
            reverse("dataset-detail", kwargs={"id": self.dataset_a.id})
        )

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_user_can_delete_only_owned_dataset(self):
        self.client.force_authenticate(self.user_b)

        forbidden_delete = self.client.delete(
            reverse("dataset-detail", kwargs={"id": self.dataset_a.id})
        )
        self.assertEqual(forbidden_delete.status_code, status.HTTP_404_NOT_FOUND)
        self.assertTrue(Dataset.objects.filter(id=self.dataset_a.id).exists())

        owned_delete = self.client.delete(
            reverse("dataset-detail", kwargs={"id": self.dataset_b.id})
        )
        self.assertEqual(owned_delete.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(Dataset.objects.filter(id=self.dataset_b.id).exists())

    def test_dataset_list_requires_authentication(self):
        response = self.client.get(reverse("dataset-list"))
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class DatasetUploadApiTests(APITestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.override = override_settings(
            MEDIA_ROOT=self.temp_dir.name,
            DATASET_LARGE_FILE_WARNING_BYTES=20,
        )
        self.override.enable()
        User = get_user_model()
        self.user = User.objects.create_user(
            username="upload_user",
            email="upload@example.com",
            password="StrongPass123!",
        )

    def tearDown(self):
        self.override.disable()
        self.temp_dir.cleanup()

    def make_csv_file(self, name="sales.csv", content=b"name,revenue\nA,100\nB,200\n"):
        return SimpleUploadedFile(name, content, content_type="text/csv")

    def test_upload_requires_authentication(self):
        response = self.client.post(
            reverse("dataset-upload"),
            {"file": self.make_csv_file(), "confirm_large_file": "true"},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_user_can_upload_csv_dataset(self):
        self.client.force_authenticate(self.user)

        response = self.client.post(
            reverse("dataset-upload"),
            {
                "file": self.make_csv_file(),
                "name": "Sales Upload",
                "description": "Local CSV upload",
                "confirm_large_file": "true",
            },
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        dataset = Dataset.objects.get(id=response.data["id"])
        self.assertEqual(dataset.owner, self.user)
        self.assertEqual(dataset.file_name, "sales.csv")
        self.assertEqual(dataset.file_type, "csv")
        self.assertEqual(dataset.row_count, 2)
        self.assertEqual(dataset.column_count, 2)
        self.assertEqual(dataset.upload_mode, Dataset.UPLOAD_MODE_LOCAL_UPLOAD)
        self.assertEqual(dataset.storage_type, Dataset.STORAGE_TYPE_LOCAL_DISK)
        self.assertEqual(dataset.status, Dataset.STATUS_READY)
        self.assertEqual(len(dataset.columns_json), 2)

    def test_upload_rejects_unsupported_extension(self):
        self.client.force_authenticate(self.user)
        file = SimpleUploadedFile("notes.txt", b"hello", content_type="text/plain")

        response = self.client.post(
            reverse("dataset-upload"),
            {"file": file},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertFalse(Dataset.objects.exists())

    def test_upload_rejects_empty_file(self):
        self.client.force_authenticate(self.user)
        file = SimpleUploadedFile("empty.csv", b"", content_type="text/csv")

        response = self.client.post(
            reverse("dataset-upload"),
            {"file": file},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertFalse(Dataset.objects.exists())

    def test_large_file_returns_warning_before_processing_without_confirmation(self):
        self.client.force_authenticate(self.user)

        response = self.client.post(
            reverse("dataset-upload"),
            {"file": self.make_csv_file(content=b"name,revenue\nA,100\nB,200\nC,300\n")},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(response.data["code"], "large_file_warning")
        self.assertFalse(Dataset.objects.exists())

    def test_large_file_upload_continues_when_confirmed(self):
        self.client.force_authenticate(self.user)

        response = self.client.post(
            reverse("dataset-upload"),
            {
                "file": self.make_csv_file(content=b"name,revenue\nA,100\nB,200\nC,300\n"),
                "confirm_large_file": "true",
            },
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Dataset.objects.count(), 1)

    def test_delete_removes_local_file_for_owned_dataset(self):
        self.client.force_authenticate(self.user)
        upload_response = self.client.post(
            reverse("dataset-upload"),
            {"file": self.make_csv_file(), "confirm_large_file": "true"},
            format="multipart",
        )
        dataset = Dataset.objects.get(id=upload_response.data["id"])
        local_path = dataset.file.path

        delete_response = self.client.delete(
            reverse("dataset-detail", kwargs={"id": dataset.id})
        )

        self.assertEqual(delete_response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(Dataset.objects.filter(id=dataset.id).exists())
        self.assertFalse(os.path.exists(local_path))

    def test_detail_returns_preview_schema_and_profile(self):
        self.client.force_authenticate(self.user)
        rows = ["customer_id,segment,purchased,notes,target"]
        for index in range(25):
            notes = "" if index == 3 else "short note"
            rows.append(f"{index},A,true,{notes},{index % 2}")
        file = self.make_csv_file(content=("\n".join(rows) + "\n").encode("utf-8"))

        upload_response = self.client.post(
            reverse("dataset-upload"),
            {"file": file, "confirm_large_file": "true"},
            format="multipart",
        )
        detail_response = self.client.get(
            reverse("dataset-detail", kwargs={"id": upload_response.data["id"]})
        )

        self.assertEqual(detail_response.status_code, status.HTTP_200_OK)
        self.assertIn("preview_json", detail_response.data)
        self.assertIn("profile_json", detail_response.data)
        self.assertEqual(len(detail_response.data["preview_json"]["rows"]), 20)
        self.assertEqual(
            detail_response.data["preview_json"]["rows"][3]["notes"],
            None,
        )
        self.assertEqual(len(detail_response.data["columns_json"]), 5)
        self.assertEqual(
            detail_response.data["columns_json"][0]["detected_type"],
            "id",
        )
        self.assertIn("quality_score", detail_response.data["profile_json"])
        self.assertIn("customer_id", detail_response.data["profile_json"]["id_like_columns"])
