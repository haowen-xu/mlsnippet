import collections
import operator
from functools import reduce

import six

from mltoolkit.utils import DocInherit, AutoInitAndCloseable

__all__ = [
    'UnsupportedOperation',
    'DataFileNotExist',
    'MetaKeyNotExist',
    'DataFSCapacity',
    'DataFS',
]


class UnsupportedOperation(Exception):
    """
    Class to indicate that a requested operation is not supported by the
    specific :class:`DataFS` subclass.
    """


class DataFileNotExist(KeyError, IOError):
    """Class to indicate a requested data file does not exist."""

    def __init__(self, filename):
        super(DataFileNotExist, self).__init__(filename)

    @property
    def filename(self):
        return self.args[0]

    def __str__(self):
        return 'Data file not exist: {!r}'.format(self.filename)


class MetaKeyNotExist(KeyError):
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


class DataFSCapacity(object):
    """
    Enumeration class to represent the capacity of a :class:`DataFS`.

    There are 7 different categories of capacities.  Every method of
    :class:`DataFS` may only work if the :class:`DataFS` has the
    particular one or more capacities.  One may check whether the
    :class:`DataFS` has a certain capacity by ``can_[capacity_name]()``.
    """

    __slots__ = ('_mode',)

    READ_DATA = 0x1
    """Can read file data, the basic capacity of a :class:`DataFS`."""

    WRITE_DATA = 0x2
    """Can write file data."""

    READ_META = 0x4
    """Can read meta data."""

    WRITE_META = 0x8
    """Can write meta data."""

    LIST_META = 0x10
    """Can enumerate the meta keys for a particular file."""

    QUICK_COUNT = 0x20
    """Can get the count of files without iterating through them."""

    RANDOM_SAMPLE = 0x40
    """Can randomly sample files without obtaining the whole file list."""

    READ_WRITE_DATA = READ_DATA | WRITE_DATA
    """Can read and write file data."""

    READ_WRITE_META = READ_META | WRITE_META
    """Can read and write meta data."""

    ALL = (READ_WRITE_DATA | READ_WRITE_META | LIST_META |
           QUICK_COUNT | RANDOM_SAMPLE)
    """All capacities are supported."""

    def __init__(self, *flags):
        """
        Construct a new :class:`DataFSCapacity`.

        Args:
            *flags: The supported capacity flags, one after another.
        """
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
        for flag in ('read_data', 'write_data', 'read_meta', 'write_meta',
                     'list_meta', 'quick_count', 'random_sample'):
            if getattr(self, 'can_{}'.format(flag))():
                pieces.append(flag)
        return '{}({})'.format(self.__class__.__name__, ','.join(pieces))


@DocInherit
class DataFS(AutoInitAndCloseable):
    """
    Base class for all data file systems.

    A :class:`DataFS` provides access to a machine learning dataset stored
    in a file system like backend.  For example, large image datasets are
    usually stored as raw image files, gathered in a directory.  Such true
    file system can be accessed by :class:`~mltoolkit.datafs.LocalFS`.

    Apart from the true file system, some may instead store these images in a
    database provided virtual file system, for example, the GridFS of MongoDB,
    which can be accessed via :class:`~mltoolkit.datafs.MongoFS`.
    """

    _buffer_size = 65536
    """The default buffer size for IO operations."""

    _initialized = False
    """Whether or not this :class:`DataFS` has been initialized?"""

    def __init__(self, strict=False):
        """
        Construct a new :class:`DataFS`.

        Args:
            strict (bool): Whether or not this :class:`DataFS` works in
                strict mode?  (default :obj:`False`)

                In strict mode, the following behaviours will take place:

                1. Accessing the value of a non-exist meta key will cause
                   a :class:`MetaKeyNotExist`, instead of getting :obj:`None`.
        """
        self._strict = strict

    @property
    def strict(self):
        """Whether or not this :class:`DataFS` works in strict mode?"""
        return self._strict

    def as_flow(self, batch_size, with_names=True, meta_keys=None,
                shuffle=False, skip_incomplete=False, names_pattern=None):
        """
        Construct a :class:`~mltoolkit.datafs.DataFSFlow`,
        a :class:`~tfsnippet.dataflow.DataFlow` driven by :meth:`iter_files`,
        :class:`iter_names` and :class:`retrieve`.

        The returned :class:`~mltoolkit.datafs.DataFSFlow` will hold a copy
        of this instance (obtained by :meth:`clone()`) instead of holding
        this instance itself.

        Args:
            batch_size (int): The mini-batch size of the returned flow.
            with_names (bool): Whether or not to include the file names
                in the returned flow? (default :obj:`True`)
            meta_keys (None or Iterable[str]): The keys of the meta data
                to be included in the returned flow. (default :obj:`None`)
            shuffle (bool): Whether or not to shuffle the files in each
                epoch of the flow?  Setting this to :obj:`True` will force
                loading the file list into memory.  (default :obj:`False`)
            skip_incomplete (bool): Whether or not to skip the final
                mini-batch, if it has fewer data than ``batch_size``?
                (default :obj:`False`, the final mini-batch will always
                 be visited even if it has fewer data than ``batch_size``)
            names_pattern (None or str or regex): The file name pattern.
                If specified, only if the file name matches this pattern,
                would the file be included in the constructed data flow.
                (default :obj:`None`)

        Returns:
            mltoolkit.datafs.DataFSFlow: A dataflow, with each mini-batch
                having numpy arrays ``([filename,] content, [meta-data...])``,
                according to the arguments.
        """
        from .dataflow import DataFSFlow
        return DataFSFlow(
            self.clone(),
            with_names=with_names,
            meta_keys=meta_keys,
            batch_size=batch_size,
            shuffle=shuffle,
            skip_incomplete=skip_incomplete,
            names_pattern=names_pattern
        )

    def as_random_flow(self, batch_size, with_names=True, meta_keys=None,
                       skip_incomplete=False):
        """
        Construct a :class:`~mltoolkit.datafs.DataFSRandomFlow`,
        a :class:`~tfsnippet.dataflow.DataFlow` driven by :meth:`random_sample`.

        The returned :class:`~mltoolkit.datafs.DataFSRandomFlow` will hold
        a copy of this instance (obtained by :meth:`clone()`) instead of
        holding this instance itself.

        Args:
            batch_size (int): The mini-batch size of the returned flow.
            with_names (bool): Whether or not to include the file names
                in the returned flow? (default :obj:`True`)
            meta_keys (None or Iterable[str]): The keys of the meta data
                to be included in the returned flow. (default :obj:`None`)
            skip_incomplete (bool): Whether or not to skip the mini-batches
                with fewer data than ``batch_size``?
                (default :obj:`False`, all the mini-batches will be visited
                 even if having fewer data than ``batch_size``)

        Returns:
            mltoolkit.datafs.DataFSRandomFlow: A dataflow, with each mini-batch
                having numpy arrays ``([filename,] content, [meta-data...])``,
                according to the arguments.

        Raises:
            UnsupportedOperation: If ``RANDOM_SAMPLE`` capacity is absent.
        """
        if not self.capacity.can_random_sample():
            raise UnsupportedOperation()
        from .dataflow import DataFSRandomFlow
        return DataFSRandomFlow(
            self.clone(),
            with_names=with_names,
            meta_keys=meta_keys,
            batch_size=batch_size,
            skip_incomplete=skip_incomplete
        )

    def clone(self):
        """
        Obtain a clone of this :class:`DataFS` instance.

        Returns:
            DataFS: The cloned :class:`DataFS`.  Only the construction
                arguments will be copied.  All the internal states
                (e.g., database connections) are kept un-initialized.
        """
        raise NotImplementedError()

    @property
    def capacity(self):
        """
        Get the capacity of this :class:`DataFS`.

        Returns:
            DataFSCapacity: The capacity object.
        """
        raise NotImplementedError()

    def count(self):
        """
        Count the files in this :class:`DataFS`.

        Will iterate through all the files via :meth:`iter_names()`, if
        ``QUICK_COUNT`` capacity is absent.

        Returns:
            int: The total number of files.
        """
        # This is a fast way to count the items in an iterator.
        # https://github.com/wbolster/cardinality/blob/master/cardinality.py#L24
        d = collections.deque(enumerate(self.iter_names(), 1), maxlen=1)
        return d[0][0] if d else 0

    def iter_names(self):
        """
        Iterate through all the file names in this :class:`DataFS`.

        Yields:
            str: The file name of each file.
        """
        raise NotImplementedError()

    def list_names(self):
        """
        Get the list of all the file names.

        Returns:
            list[str]: The file names list.
        """
        return list(self.iter_names())

    def sample_names(self, n_samples):
        """
        Sample ``n_samples`` file names from this :class:`DataFS`.

        Args:
            n_samples (int): Number of names to sample.
                The returned names may be fewer than this number,
                if there are less than ``n_samples`` files in this
                :class:`DataFS`.

        Returns:
            list[str]: The list of sampled file names.

        Raises:
            UnsupportedOperation: If ``RANDOM_SAMPLE`` capacity is absent.
        """
        raise NotImplementedError()

    def iter_files(self, meta_keys=None):
        """
        Iterate through all the files in this :class:`DataFS`.

        Args:
            meta_keys (None or Iterable[str]): The keys of the meta data
                to be retrieved. (default :obj:`None`)

        Yields:
            (filename, content, [meta-data...]): A tuple containing the
                name of a file, its content, and the values of each meta
                data corresponding to ``meta_keys``.  If a requested key
                is absent for a file, :obj:`None` will take the place.

        Raises:
            UnsupportedOperation: If ``meta_keys`` is specified, but
                ``READ_META`` capacity is absent.
        """
        meta_keys = tuple(meta_keys or ())
        for name in self.iter_names():
            yield (name,) + self.retrieve(name, meta_keys)

    def sample_files(self, n_samples, meta_keys=None):
        """
        Sample ``n_samples`` files from this :class:`DataFS`.

        Args:
            n_samples (int): The number of files to sample.
            meta_keys (None or Iterable[str]): The keys of the meta data
                to be retrieved. (default :obj:`None`)

        Returns:
            list[(filename, content, [meta-data...])]: A list of tuples,
                each tuple contains the name of a file, its content, and
                the values of each meta data corresponding to ``meta_keys``.
                If a requested key is absent for a file, :obj:`None` will
                take the place.

        Raises:
            UnsupportedOperation: If ``meta_keys`` is specified, but
                ``READ_META`` capacity is absent.
        """
        meta_keys = tuple(meta_keys or ())
        names = self.sample_names(n_samples)
        return [(name,) + self.retrieve(name, meta_keys) for name in names]

    def retrieve(self, filename, meta_keys=None):
        """
        Retrieve the content and maybe meta data of a file.

        Args:
            filename (str): The name of the file to be retrieved.
            meta_keys (None or Iterable[str]): The keys of the meta data
                to be retrieved. (default :obj:`None`)

        Returns:
            bytes or (bytes, [meta-data...]): The content, or a tuple
                containing the content and the meta values, corresponding
                to ``meta_keys``.  If a requested key is absent for a file,
                :obj:`None` will take the place.

        Notes:
            As long as ``meta_keys`` is not None, a tuple will always
            be returned, even if ``meta_keys`` is an empty collection.

        Raises:
            UnsupportedOperation: If ``meta_keys`` is specified, but
                ``READ_META`` capacity is absent.
        """
        if meta_keys is not None:
            meta_keys = tuple(meta_keys)
            return (self.get_data(filename),) + \
                (meta_keys and self.get_meta(filename, meta_keys))
        else:
            return self.get_data(filename)

    def get_data(self, filename):
        """
        Get the content of a file.

        Args:
            filename (str): The name of the file.

        Returns:
            bytes: The content of a file.
        """
        return self.retrieve(filename)

    def put_data(self, filename, data):
        """
        Save the content of a file.

        Args:
            filename (str): The name of the file.
            data (bytes or file-like): The content of the file,
                or a file-like object with ``read(size)`` method.

        Raises:
            UnsupportedOperation: If ``WRITE_DATA`` capacity is absent.
        """
        if isinstance(data, six.binary_type):
            with self.open(filename, 'w') as f:
                f.write(data)
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
        """
        Open a file-like object to read / write a file.

        Args:
            filename (str): The name of the file.
            mode ({'r', 'w'}): The open mode of the file, either 'r' for
                reading or 'w' for writing.  Other modes are not supported
                in general.

        Returns:
            file-like: The file-like object.  This object will be immediately
                closed as soon as this :class:`DataFS` instance is closed.

        Raises:
            UnsupportedOperation: If the specified mode is not supported,
                e.g., ``mode == 'w'`` but ``WRITE_DATA`` capacity is absent.

        """
        raise NotImplementedError()

    def isfile(self, filename):
        """
        Check whether or not a file exists.

        Args:
            filename (str): The name of the file.

        Returns:
            bool: :obj:`True` if ``filename`` exists and is a file,
                and :obj:`False` otherwise.
        """
        raise NotImplementedError()

    def batch_isfile(self, filenames):
        """
        Check whether or not the files exist.

        Args:
            filenames (Iterable[str]): The names of the files.

        Returns:
            list[bool]: A list of indicators, where :obj:`True` if the
                corresponding ``filename`` exists and is a file, and
                :obj:`False` otherwise.
        """
        return [self.isfile(filename) for filename in filenames]

    def list_meta(self, filename):
        """
        List the meta keys of a file.

        Args:
            filename (str): The name of the file.

        Returns:
            list[str]: The keys of the meta data of the file.

        Raises:
            UnsupportedOperation: If the ``LIST_META`` capacity is absent.
        """
        raise NotImplementedError()

    def get_meta(self, filename, meta_keys):
        """
        Get meta data of a file.

        Args:
            filename (str): The name of the file.
            meta_keys (Iterable[str]): The keys of the meta data.

        Returns:
            list[any]: The meta values, corresponding to ``meta_keys``.
                If a requested key is absent for a file, :obj:`None` will
                take the place.

        Raises:
            UnsupportedOperation: If the ``READ_META`` capacity is absent.
        """
        raise NotImplementedError()

    def batch_get_meta(self, filenames, meta_keys):
        """
        Get meta data of files.

        Args:
            filenames (Iterable[str]): The names of the files.
            meta_keys (Iterable[str]): The keys of the meta data.

        Returns:
            list[tuple[any] or None]: A list of meta values, or :obj:`None`
                if the corresponding file does not exist.
        """
        if meta_keys is not None:
            meta_keys = tuple(meta_keys)
        return [self.get_meta(name, meta_keys) for name in filenames]

    def get_meta_dict(self, filename):
        """
        Get all the meta data of a file, as a dict.

        Args:
            filename (str): The name of the file.

        Returns:
            dict[str, any]: The meta values, as a dict.

        Raises:
            UnsupportedOperation: If the ``READ_META`` or ``LIST_META``
                capacity is absent.
        """
        meta_keys = self.list_meta(filename)
        meta_values = self.get_meta(filename, meta_keys)
        return {k: v for k, v in zip(meta_keys, meta_values)}

    def put_meta(self, filename, meta_dict=None, **meta_dict_kwargs):
        """
        Update the meta data of a file.  The un-mentioned meta data will
        remain unchanged.  This method is not necessarily faster than
        :meth:`clear_and_put_meta`.  In some backends it may be implemented
        by first calling :class:`get_meta_dict`, then updating the meta dict
        in memory, and finally calling :class:`clear_and_put_meta`.

        Args:
            filename (str): The name of the file.
            meta_dict (dict[str, any]): The meta values to be updated.
            **meta_dict_kwargs: The meta values to be updated, as keyword
                arguments.  This will override the values provided in
                ``meta_dict``.

        Raises:
            UnsupportedOperation: If the ``WRITE_META`` capacity (and
                possibly the ``READ_META`` capacity) is(are) absent.
        """
        raise NotImplementedError()

    def clear_and_put_meta(self, filename, meta_dict=None, **meta_dict_kwargs):
        """
        Set the meta data of a file.  The un-mentioned meta data will be
        cleared.  This method is not necessarily slower than :meth:`put_meta`.

        Args:
            filename (str): The name of the file.
            meta_dict (dict[str, any]): The meta values to be updated.
            **meta_dict_kwargs: The meta values to be updated, as keyword
                arguments.  This will override the values provided in
                ``meta_dict``.

        Raises:
            UnsupportedOperation: If the ``WRITE_META`` capacity is absent.
        """
        self.clear_meta(filename)
        self.put_meta(filename, meta_dict, **meta_dict_kwargs)

    def clear_meta(self, filename):
        """
        Clear all the meta data of a file.

        Args:
            filename (str): The name of the file.

        Raises:
            UnsupportedOperation: If the ``WRITE_META`` capacity (and
                possibly the ``LIST_META`` capacity) is(are) absent.
        """
        raise NotImplementedError()
