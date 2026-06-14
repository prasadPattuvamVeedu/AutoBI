from django.core.exceptions import PermissionDenied


def is_owner(user, obj, owner_field="owner"):
    return getattr(obj, owner_field, None) == user


def require_owner(user, obj, owner_field="owner"):
    if not is_owner(user, obj, owner_field):
        raise PermissionDenied("You do not have permission to access this object.")
    return obj
