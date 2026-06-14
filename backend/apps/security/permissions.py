from rest_framework.permissions import BasePermission


class IsOwner(BasePermission):
    """Allow access only when an object belongs to request.user."""

    message = "You do not have permission to access this object."

    def has_object_permission(self, request, view, obj):
        return getattr(obj, "owner", None) == request.user
