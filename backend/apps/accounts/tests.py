from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from unittest.mock import patch


class AuthApiTests(APITestCase):
    def test_register_login_profile_refresh_and_logout(self):
        register_response = self.client.post(
            reverse("register"),
            {
                "username": "day2user",
                "email": "day2@example.com",
                "password": "StrongPass123!",
                "confirm_password": "StrongPass123!",
            },
            format="json",
        )

        self.assertEqual(register_response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(get_user_model().objects.filter(username="day2user").exists())

        login_response = self.client.post(
            reverse("token_obtain_pair"),
            {"username": "day2user", "password": "StrongPass123!"},
            format="json",
        )

        self.assertEqual(login_response.status_code, status.HTTP_200_OK)
        access = login_response.data["access"]
        refresh = login_response.data["refresh"]

        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {access}")
        profile_response = self.client.get(reverse("profile"))
        self.assertEqual(profile_response.status_code, status.HTTP_200_OK)
        self.assertEqual(profile_response.data["username"], "day2user")

        refresh_response = self.client.post(
            reverse("token_refresh"),
            {"refresh": refresh},
            format="json",
        )
        self.assertEqual(refresh_response.status_code, status.HTTP_200_OK)
        self.assertIn("access", refresh_response.data)

        logout_response = self.client.post(
            reverse("logout"),
            {"refresh": refresh},
            format="json",
        )
        self.assertEqual(logout_response.status_code, status.HTTP_200_OK)

        blacklisted_refresh_response = self.client.post(
            reverse("token_refresh"),
            {"refresh": refresh},
            format="json",
        )
        self.assertEqual(blacklisted_refresh_response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_profile_requires_authentication(self):
        response = self.client.get(reverse("profile"))
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    @override_settings(GOOGLE_CLIENT_ID="test-google-client-id")
    @patch("apps.accounts.views.id_token.verify_oauth2_token")
    def test_google_login_creates_user_and_returns_tokens(self, mock_verify):
        mock_verify.return_value = {
            "email": "googleuser@example.com",
            "given_name": "Google",
            "family_name": "User",
        }

        response = self.client.post(
            reverse("google_auth"),
            {"credential": "GOOGLE_ID_TOKEN"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("access", response.data)
        self.assertIn("refresh", response.data)
        self.assertEqual(response.data["user"]["email"], "googleuser@example.com")
        self.assertTrue(
            get_user_model().objects.filter(email="googleuser@example.com").exists()
        )

    @override_settings(GOOGLE_CLIENT_ID="test-google-client-id")
    @patch("apps.accounts.views.id_token.verify_oauth2_token")
    def test_google_login_reuses_user_with_same_email(self, mock_verify):
        user = get_user_model().objects.create_user(
            username="existing",
            email="existing@example.com",
        )
        mock_verify.return_value = {
            "email": "existing@example.com",
            "given_name": "Existing",
            "family_name": "User",
        }

        response = self.client.post(
            reverse("google_auth"),
            {"credential": "GOOGLE_ID_TOKEN"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["user"]["id"], user.id)
        self.assertEqual(
            get_user_model().objects.filter(email="existing@example.com").count(),
            1,
        )
