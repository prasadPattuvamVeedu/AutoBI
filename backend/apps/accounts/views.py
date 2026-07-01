from django.conf import settings
from django.contrib.auth import get_user_model
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework import status
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import RefreshToken

from .serializers import RegisterSerializer, UserSerializer


User = get_user_model()


def build_unique_username(email):
    base_username = email.split("@")[0] or "google_user"
    username = base_username
    counter = 1

    while User.objects.filter(username=username).exists():
        counter += 1
        username = f"{base_username}{counter}"

    return username


class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response(
            {
                "message": "User registered successfully.",
                "user": UserSerializer(user).data,
            },
            status=status.HTTP_201_CREATED,
        )


class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)


class GoogleAuthView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        # Google Identity Services returns an ID token in `credential`.
        # Keep token/access_token aliases so older frontend builds do not fail.
        credential = (
            request.data.get("credential")
            or request.data.get("token")
            or request.data.get("id_token")
            or request.data.get("access_token")
        )

        if isinstance(credential, str):
            credential = credential.strip()

        if not credential:
            return Response(
                {"credential": "This field is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        configured_client_id = (settings.GOOGLE_CLIENT_ID or "").strip()
        if not configured_client_id:
            return Response(
                {"detail": "GOOGLE_CLIENT_ID is missing from backend configuration."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            token_info = id_token.verify_oauth2_token(
                credential,
                google_requests.Request(),
                configured_client_id,
                clock_skew_in_seconds=10,
            )
        except ValueError as exc:
            response_data = {
                "credential": "Invalid Google credential.",
                "hint": (
                    "Check that frontend VITE_GOOGLE_CLIENT_ID and backend "
                    "GOOGLE_CLIENT_ID are the same OAuth Web Client ID, and that "
                    "http://localhost:5173 is added to Authorized JavaScript origins."
                ),
            }
            if settings.DEBUG:
                response_data["debug"] = str(exc)
            return Response(response_data, status=status.HTTP_400_BAD_REQUEST)

        email = token_info.get("email")
        if not email:
            return Response(
                {"credential": "Google credential did not include an email."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user = User.objects.filter(email__iexact=email).first()
        if not user:
            user = User.objects.create_user(
                username=build_unique_username(email),
                email=email,
                first_name=token_info.get("given_name", ""),
                last_name=token_info.get("family_name", ""),
            )
        else:
            user.first_name = user.first_name or token_info.get("given_name", "")
            user.last_name = user.last_name or token_info.get("family_name", "")
            user.save(update_fields=["first_name", "last_name"])

        refresh = RefreshToken.for_user(user)

        return Response(
            {
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "user": UserSerializer(user).data,
            },
            status=status.HTTP_200_OK,
        )


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        refresh_token = request.data.get("refresh")
        if not refresh_token:
            return Response(
                {"refresh": "This field is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            RefreshToken(refresh_token).blacklist()
        except TokenError:
            return Response(
                {"refresh": "Token is invalid or already blacklisted."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response({"message": "Logged out successfully."}, status=status.HTTP_200_OK)
