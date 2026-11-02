class StorageConfigurationError(ValueError):
    """The storage engine rejected the configuration: unknown bucket or organization, invalid credentials"""


class StorageUnavailableError(ConnectionError):
    """The storage engine cannot be reached (network error, server down or timeout)"""


class StorageRejectedError(ValueError):
    """The storage engine refused the data itself (e.g. a field whose type conflicts with the stored one): retrying
    the same rows cannot succeed"""
