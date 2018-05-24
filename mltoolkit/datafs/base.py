import collections
import operator
from functools import reduce

import six

from mltoolkit.utils import DocInherit

__all__ = [
    'DataFileInfo',
    'UnsupportedOperation',
    'DataFSCapacity',
    'DataFS',
]


class DataFileInfo(object):

    __slots__ = ('_filename', '_size', '_meta')

    def __init__(self, filename, size, meta=None):
        self._filename = filename
        self._size = size
        self._meta = meta

    @property
    def filename(self):
        return self._filename

    @property
    def size(self):
        return self._size

    @property
    def meta(self):
        return self._meta


class UnsupportedOperation(Exception):
    """
    Class to indicate that a requested operation is not supported by the
    specific :class:`DataFS` subclass.
    """


class DataFSCapacity(object):

    __slots__ = ('_mode',)

    READ_DATA = 0x1
    WRITE_DATA = 0x2
    QUICK_COUNT = 0x4
    RANDOM_SAMPLE = 0x8

    def __init__(self, *flags):
        self._mode = reduce(operator.or_, flags, 0)

    def can_read_data(self):
        return not not (self._mode & self.READ_DATA)

    def can_write_data(self):
        return not not (self._mode & self.WRITE_DATA)

    def can_quick_count(self):
        return not not (self._mode & self.QUICK_COUNT)

    def can_random_sample(self):
        return not not (self._mode & self.RANDOM_SAMPLE)

    def __repr__(self):
        pieces = []
        for m in dir(self):
            if m.startswith('can_'):
                v = getattr(self, m)()
                if v:
                    pieces.append(m)
        return '{}({})'.format(self.__class__.__name__, ','.join(pieces))


@DocInherit
class DataFS(object):
    """
    Base class for all dataset file systems.
    """

    _buffer_size = 65536
    _initialized = False

    def as_flow(self, batch_size, with_names=False, shuffle=False,
                skip_incomplete=False):
        from .dataflow import DataFSFlow
        return DataFSFlow(
            self.clone(), with_names=with_names, batch_size=batch_size,
            shuffle=shuffle, skip_incomplete=skip_incomplete
        )

    def clone(self):
        raise NotImplementedError()

    def _init(self):
        raise NotImplementedError()

    def init(self):
        if not self._initialized:
            self._init()
            self._initialized = True

    def _destroy(self):
        raise NotImplementedError()

    def destroy(self):
        if self._initialized:
            try:
                self._destroy()
            finally:
                self._initialized = False

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

    @property
    def capacity(self):
        raise NotImplementedError()

    def count(self):
        # This is a fast way to count the items in an iterator.
        # https://github.com/wbolster/cardinality/blob/master/cardinality.py#L24
        d = collections.deque(enumerate(self.iter_names(), 1), maxlen=1)
        return d[0][0] if d else 0

    def iter_names(self):
        raise NotImplementedError()

    def list_names(self):
        return list(self.iter_names())

    def sample_names(self, n_samples):
        raise NotImplementedError()

    def iter_files(self):
        for name in self.iter_names():
            yield name, self.retrieve(name)

    def sample_files(self, n_samples):
        names = self.sample_names(n_samples)
        return [(name, self.retrieve(name)) for name in names]

    def retrieve(self, filename):
        with self.open(filename, 'r') as f:
            return f.read()

    def upload(self, filename, data):
        if isinstance(data, six.binary_type):
            with self.open(filename, 'w') as f:
                f.write(data)
        elif hasattr(data, 'readinto'):
            buf = bytearray(self._buffer_size)
            with self.open(filename, 'w') as f:
                while True:
                    k = data.readinto(buf)
                    if k <= 0:
                        break
                    f.write(buf[:k])
        elif hasattr(data, 'read'):
            with self.open(filename, 'w') as f:
                while True:
                    buf = data.read(self._buffer_size)
                    if not buf:
                        break
                    f.write(buf)
        else:
            raise TypeError('`data` must be bytes or a file-like object.')

    def open(self, filename, mode):
        raise NotImplementedError()
