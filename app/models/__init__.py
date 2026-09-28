"""ORM entities. Importing this package registers every model on ``Base.metadata``."""

from app.models.meeting import Meeting

__all__ = ["Meeting"]
