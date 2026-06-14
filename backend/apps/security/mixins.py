class OwnerQuerySetMixin:
    """Filter querysets to objects owned by the authenticated user."""

    owner_field = "owner"

    def get_queryset(self):
        queryset = super().get_queryset()
        return queryset.filter(**{self.owner_field: self.request.user})


class PerformCreateWithOwnerMixin:
    """Save new objects with request.user as owner."""

    owner_field = "owner"

    def perform_create(self, serializer):
        serializer.save(**{self.owner_field: self.request.user})


class UserOwnedObjectMixin(OwnerQuerySetMixin, PerformCreateWithOwnerMixin):
    """Convenience mixin for future request.user-owned models."""
