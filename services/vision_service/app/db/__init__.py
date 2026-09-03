"""Capa de persistencia: motor, sesión y modelos ORM."""

from .base import Base
from .session import get_session, session_scope

__all__ = ["Base", "get_session", "session_scope"]
