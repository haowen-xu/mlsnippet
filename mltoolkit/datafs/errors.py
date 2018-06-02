__all__ = [
    'DataFSError', 'UnsupportedOperation', 'InvalidOpenMode',
    'DataFileNotExist', 'MetaKeyNotExist',
]


class DataFSError(Exception):
    """Base class for all :class:`DataFS` errors."""


class UnsupportedOperation(DataFSError):
    """
    Class to indicate that a requested operation is not supported by the
    specific :class:`DataFS` subclass.
    """


class InvalidOpenMode(UnsupportedOperation):
    """
    Class to indicate that the specified open mode is not supported.
    """

    def __init__(self, mode):
        super(InvalidOpenMode, self).__init__(mode)

    @property
    def mode(self):
        return self.args[0]

    def __str__(self):
        return 'Invalid open mode: {!r}'.format(self.mode)


class DataFileNotExist(DataFSError):
    """Class to indicate a requested data file does not exist."""

    def __init__(self, filename):
        super(DataFileNotExist, self).__init__(filename)

    @property
    def filename(self):
        return self.args[0]

    def __str__(self):
        return 'Data file not exist: {!r}'.format(self.filename)


class MetaKeyNotExist(DataFSError):
    """Class to indicate a requested meta key does not exist."""

    def __init__(self, filename, meta_key):
        super(MetaKeyNotExist, self).__init__(filename, meta_key)

    @property
    def filename(self):
        return self.args[0]

    @property
    def meta_key(self):
        return self.args[1]

    def __str__(self):
        return 'In file {!r}: meta key not exist: {!r}'. \
            format(self.filename, self.meta_key)
