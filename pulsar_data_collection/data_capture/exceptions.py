class CustomExceptionWithMessage(Exception):
    """
    Custom error that is raised in DataCapture submodule
    with FieldValue and Custom Message passed as input
    """

    def __init__(self, value: str, message: str) -> None:
        self.value = value
        self.message = message
        super().__init__(message)
