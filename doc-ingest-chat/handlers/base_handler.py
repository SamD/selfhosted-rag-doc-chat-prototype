from abc import ABC, abstractmethod
from typing import Generator, Optional


class BaseContentTypeHandler(ABC):
    """
    Abstract base class for content type handlers using the Chain of Responsibility pattern.
    """

    def __init__(self, next_handler: Optional["BaseContentTypeHandler"] = None):
        self._next_handler = next_handler

    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """
        Check if the handler can process the given file type.
        """
        pass

    @abstractmethod
    def stream_content(self, file_path: str) -> Generator[str, None, None]:
        """
        Extract and yield content from the file as a stream of strings.
        """
        pass

    def handle(self, file_path: str) -> Optional[Generator[str, None, None]]:
        """
        Traverse the chain of responsibility to find the correct handler and return the stream.
        """
        if self.can_handle(file_path):
            return self.stream_content(file_path)
        elif self._next_handler:
            return self._next_handler.handle(file_path)
        return None
