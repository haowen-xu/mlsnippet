import collections
import operator
from functools import reduce

import six

from mltoolkit.utils import DocInherit, AutoInitAndCloseable

__all__ = [
    'UnsupportedOperation',
    'DataFSCapacity',
    'DataFS',
]


class UnsupportedOperation(Exception):
    """
    Class to indicate that a requested operation is not supported by the
    specific :class:`DataFS` subclass.
    """


class DataFSCapacity(object):

    __slots__ = ('_mode',)

    READ_DATA = 0x1
    WRITE_DATA = 0x2
    READ_META = 0x4
    WRITE_META = 0x8
    LIST_META = 0x10
    QUICK_COUNT = 0x20
    RANDOM_SAMPLE = 0x40
    READ_WRITE_DATA = READ_DATA | WRITE_DATA
    READ_WRITE_META = READ_META | WRITE_META
    ALL = (READ_WRITE_DATA | READ_WRITE_META | LIST_META |
           QUICK_COUNT | RANDOM_SAMPLE)

    def __init__(self, *flags):
        self._mode = reduce(operator.or_, flags, 0)

    def can_read_data(self):
        return (self._mode & self.READ_DATA) != 0

    def can_write_data(self):
        return (self._mode & self.WRITE_DATA) != 0

    def can_read_meta(self):
        return (self._mode & self.READ_META) != 0

    def can_write_meta(self):
        return (self._mode & self.WRITE_META) != 0

    def can_list_meta(self):
        return (self._mode & self.LIST_META) != 0

    def can_quick_count(self):
        return (self._mode & self.QUICK_COUNT) != 0

    def can_random_sample(self):
        return (self._mode & self.RANDOM_SAMPLE) != 0

    def __repr__(self):
        pieces = []
        for m in dir(self):
            if m.startswith('can_'):
                v = getattr(self, m)()
                if v:
                    pieces.append(m)
        return '{}({})'.format(self.__class__.__name__, ','.join(pieces))


@DocInherit
class DataFS(AutoInitAndCloseable):
    """
    Base class for all dataset file systems.
    """

    _buffer_size = 65536
    _initialized = False

    def as_flow(self, batch_size, with_names=False, meta_keys=None,
                shuffle=False, skip_incomplete=False):
        from .dataflow import DataFSFlow
        return DataFSFlow(
            self.clone(),
            with_names=with_names,
            meta_keys=meta_keys,
            batch_size=batch_size,
            shuffle=shuffle,
            skip_incomplete=skip_incomplete
        )

    def clone(self):
        raise NotImplementedError()

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

    def iter_files(self, meta_keys=None):
        if meta_keys is not None:
            meta_keys = tuple(meta_keys)
        for name in self.iter_names():
            yield (name,) + self.retrieve(name, meta_keys)

    def sample_files(self, n_samples, meta_keys=None):
        if meta_keys is not None:
            meta_keys = tuple(meta_keys)
        names = self.sample_names(n_samples)
        return [(name,) + self.retrieve(name, meta_keys) for name in names]

    def retrieve(self, filename, meta_keys=None):
        if meta_keys is not None:
            meta_keys = tuple(meta_keys)
            return (self.get_data(filename),) + \
                (meta_keys and self.get_meta(filename, meta_keys))
        else:
            return self.get_data(filename)

    def get_data(self, filename):
        with self.open(filename, 'r') as f:
            return f.read()

    def put_data(self, filename, data):
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

    def list_meta(self, filename):
        raise NotImplementedError()

    def get_meta(self, filename, meta_keys):
        raise NotImplementedError()

    def batch_get_meta(self, filenames, meta_keys):
        if meta_keys is not None:
            meta_keys = tuple(meta_keys)
        return [self.get_meta(name, meta_keys) for name in filenames]

    def get_meta_dict(self, filename):
        if meta_keys is not None:
            meta_keys = tuple(meta_keys)
        meta_keys = self.list_meta(filename)
        meta_values = self.get_meta(filename, meta_keys)
        return {k: v for k, v in zip(meta_keys, meta_values)}

    def put_meta(self, filename, meta_dict=None, **meta_dict_kwargs):
        raise NotImplementedError()

    def clear_and_put_meta(self, filename, meta_dict=None, **meta_dict_kwargs):
        self.clear_meta(filename)
        self.put_meta(filename, meta_dict, **meta_dict_kwargs)

    def clear_meta(self, filename):
        raise NotImplementedError()
