class StorageConfigurationError(ValueError):
    """The storage engine rejected the configuration: unknown bucket or organization, invalid credentials"""


class StorageUnavailableError(ConnectionError):
    """The storage engine cannot be reached (network error, server down or timeout)"""
