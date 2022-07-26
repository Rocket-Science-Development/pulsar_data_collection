class StorageEngineDoesntExist(Exception):
    """Custom error that is raised when the DB conenctor doesn't exist"""

    def __init__(self, value: str, message: str) -> None:
        self.value = value
        self.message = message
        super().__init__(message)
