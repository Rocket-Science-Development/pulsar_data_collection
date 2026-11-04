from typing import Optional


class StorageConfigurationError(ValueError):
    """The storage engine rejected the configuration: unknown bucket or organization, invalid credentials"""


class StorageUnavailableError(ConnectionError):
    """The storage engine cannot be reached (network error, server down or timeout)"""


class StorageWriteUncertainError(StorageUnavailableError):
    """The request was sent but its answer never came: the storage may or may not have stored the rows

    Writes are idempotent (same measurement, tags and timestamps overwrite the same points), so these rows can be
    spooled and written again without creating duplicates. Without spool they are counted in `uncertain_rows`.
    """


class StorageWriteTimeoutError(StorageWriteUncertainError):
    """The request was sent but no answer came back in time: the storage may or may not have stored the rows"""


class StorageConnectionResetError(StorageWriteUncertainError):
    """The connection was reset or closed once the request was (being) sent, before the answer: the storage may or
    may not have stored the rows"""


class StorageRejectedError(ValueError):
    """The storage engine refused the data itself (e.g. a field whose type conflicts with the stored one): retrying
    the same rows cannot succeed

    `dropped`: number of rows the storage reports as refused when it wrote the others (InfluxDB partial write, HTTP 422
    `dropped=N`); None when it is not known (the whole request may have been refused).
    """

    def __init__(self, *args, dropped: Optional[int] = None):
        super().__init__(*args)
        self.dropped = dropped
