"""Language -> ASR backend routing.

Configured by ``ASR_LANGUAGE_BACKENDS`` (``en:whisper,hi:whisper,or:indic``).
``select`` is called once per meeting today; Prompt 6 will call it per segment
after language identification, which is why routing is a separate object.
"""

from collections.abc import Mapping

from app.core.exceptions import UnsupportedLanguageError
from app.services.asr.base import ASRBackend


class ASRRouter:
    def __init__(
        self,
        backends: Mapping[str, ASRBackend],
        language_backends: Mapping[str, str],
        auto_backend: str = "whisper",
    ) -> None:
        unknown = {name for name in language_backends.values() if name not in backends}
        if auto_backend not in backends:
            unknown.add(auto_backend)
        if unknown:
            raise ValueError(
                f"ASR_LANGUAGE_BACKENDS references unknown backend(s) {sorted(unknown)}; "
                f"available: {sorted(backends)}"
            )
        self.backends = dict(backends)
        self.language_backends = dict(language_backends)
        self.auto_backend = auto_backend

    @property
    def languages(self) -> frozenset[str]:
        return frozenset(self.language_backends)

    def select(self, language: str | None) -> ASRBackend:
        """Backend for ``language``; ``None`` means automatic language detection."""
        if language is None:
            backend = self.backends[self.auto_backend]
            if not backend.supports_auto_detect:
                raise UnsupportedLanguageError(
                    f"Backend '{backend.name}' cannot auto-detect the language; "
                    "pass a language hint on upload.",
                    details={"backend": backend.name},
                )
            return backend

        language = language.lower()
        name = self.language_backends.get(language)
        if name is None:
            raise UnsupportedLanguageError(
                f"No ASR backend is configured for language '{language}'. "
                "Add it to ASR_LANGUAGE_BACKENDS (e.g. 'or:indic').",
                details={"language": language, "configured": sorted(self.language_backends)},
            )
        backend = self.backends[name]
        if language not in backend.supported_languages:
            raise UnsupportedLanguageError(
                f"ASR backend '{name}' does not support language '{language}'. "
                f"It supports: {', '.join(sorted(backend.supported_languages))}.",
                details={"language": language, "backend": name},
            )
        return backend
