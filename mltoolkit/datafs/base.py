import collections
import re

import six

from mltoolkit.utils import DocInherit, AutoInitAndCloseable
from .errors import UnsupportedOperation, DataFileNotExist
from .utils import maybe_close

__all__ = [
    'DataFSCapacity',
    'DataFS',
]


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

    def __init__(self, mode=0):
        """
        Construct a new :class:`DataFSCapacity`.

        Args:
            mode (int): The mode number of this capacity flag.
        """
        if isinstance(mode, DataFSCapacity):
            mode = mode._mode
        self._mode = int(mode)

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

    def __eq__(self, other):
        return isinstance(other, DataFSCapacity) and self._mode == other._mode

    def __hash__(self):
        return hash(self._mode)


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

    def __init__(self, capacity, strict=False):
        """
        Initialize the base :class:`DataFS` class.

        Args:
            capacity (int or DataFSCapacity): Specify the capacity of the
                derived :class:`DataFS`.
            strict (bool): Whether or not this :class:`DataFS` works in
                strict mode?  (default :obj:`False`)

                In strict mode, the following behaviours will take place:

                1. Accessing the value of a non-exist meta key will cause
                   a :class:`MetaKeyNotExist`, instead of getting :obj:`None`.
        """
        self._capacity = DataFSCapacity(capacity)
        self._strict = strict

    @property
    def capacity(self):
        """
        Get the capacity of this :class:`DataFS`.

        Returns:
            DataFSCapacity: The capacity object.
        """
        return self._capacity

    @property
    def strict(self):
        """Whether or not this :class:`DataFS` works in strict mode?"""
        return self._strict

    def as_flow(self, batch_size, with_names=True, meta_keys=None,
                shuffle=False, skip_incomplete=False, names_pattern=None):
        """
        Construct a :class:`~tfsnippet.dataflow.DataFlow`, which iterates
        through the files once and only once in an epoch.

        The returned :class:`~mltoolkit.datafs.DataFSFlow` will hold a copy
        of this instance (obtained by :meth:`clone()`) instead of holding
        this instance itself.

        Args:
            batch_size (int): Size of each mini-batch.
            with_names (bool): Whether or not to include the file names
                in the returned flow? (default :obj:`True`)
            meta_keys (None or Iterable[str]): The keys of the meta data
                to be included in the returned flow. (default :obj:`None`)
            shuffle (bool): Whether or not to shuffle the files in each
                epoch of the flow?  Setting this to :obj:`True` will force
                loading the file list into memory.  (default :obj:`False`)
            skip_incomplete (bool): Whether or not to exclude a mini-batch,
                if it has fewer data than ``batch_size``?
                (default :obj:`False`, the final mini-batch will always
                 be visited even if it has fewer data than ``batch_size``)
            names_pattern (None or str or regex): The file name pattern.
                If specified, only if the file name matches this pattern,
                would the file be included in the constructed data flow.
                Specifying this option will force loading the file list
                into memory. (default :obj:`None`)

        Returns:
            tfsnippet.dataflow.DataFlow: A dataflow, with each mini-batch
                having numpy arrays ``([filename,] content, [meta-data...])``,
                according to the arguments.
        """
        from .dataflow import DataFSForwardFlow, DataFSIndexedFlow

        # quick path: use forward flow if no shuffling and name filtering
        if not shuffle and names_pattern is None:
            return DataFSForwardFlow(
                fs=self.clone(),
                batch_size=batch_size,
                with_names=with_names,
                meta_keys=meta_keys,
                skip_incomplete=skip_incomplete,
            )

        # slow path: load the names, then do filtering if required,
        # and use indexed flow to serve
        else:
            if names_pattern is None:
                names = self.list_names()
            else:
                names_pattern = re.compile(names_pattern)
                names = [n for n in self.iter_names() if names_pattern.match(n)]
            return DataFSIndexedFlow(
                fs=self.clone(),
                names=names,
                batch_size=batch_size,
                with_names=with_names,
                meta_keys=meta_keys,
                shuffle=shuffle,
                skip_incomplete=skip_incomplete,
            )

    def sub_flow(self, batch_size, names, with_names=True, meta_keys=None,
                 shuffle=False, skip_incomplete=False):
        """
        Construct a :class:`~tfsnippet.dataflow.DataFlow`, which iterates
        through the files according to selected `names`.

        The returned :class:`~mltoolkit.datafs.DataFSFlow` will hold a copy
        of this instance (obtained by :meth:`clone()`) instead of holding
        this instance itself.

        Args:
            batch_size (int): Size of each mini-batch.
            names (list[str]): The names of files to retrieve.
            with_names (bool): Whether or not to include the file names
                in the returned flow? (default :obj:`True`)
            meta_keys (None or Iterable[str]): The keys of the meta data
                to be included in the returned flow. (default :obj:`None`)
            shuffle (bool): Whether or not to shuffle the files in each
                epoch of the flow?  Setting this to :obj:`True` will force
                loading the file list into memory.  (default :obj:`False`)
            skip_incomplete (bool): Whether or not to exclude a mini-batch,
                if it has fewer data than ``batch_size``?
                (default :obj:`False`, the final mini-batch will always
                 be visited even if it has fewer data than ``batch_size``)

        Returns:
            tfsnippet.dataflow.DataFlow: A dataflow, with each mini-batch
                having numpy arrays ``([filename,] content, [meta-data...])``,
                according to the arguments.
        """
        from .dataflow import DataFSIndexedFlow
        return DataFSIndexedFlow(
            fs=self.clone(),
            names=names,
            batch_size=batch_size,
            with_names=with_names,
            meta_keys=meta_keys,
            shuffle=shuffle,
            skip_incomplete=skip_incomplete,
        )

    def random_flow(self, batch_size, with_names=True, meta_keys=None,
                    skip_incomplete=False, batch_count=None):
        """
        Construct a :class:`~tfsnippet.dataflow.DataFlow`, with infinite
        or pre-configured number of mini-batches in an epoch, randomly
        sampled from the whole :class:`DataFS`.

        The returned :class:`~mltoolkit.datafs.DataFSRandomFlow` will hold
        a copy of this instance (obtained by :meth:`clone()`) instead of
        holding this instance itself.

        Args:
            batch_size (int): Size of each mini-batch.
            with_names (bool): Whether or not to include the file names
                in the returned flow? (default :obj:`True`)
            meta_keys (None or Iterable[str]): The keys of the meta data
                to be included in the returned flow. (default :obj:`None`)
            skip_incomplete (bool): Whether or not to exclude a mini-batch,
                if it has fewer data than ``batch_size``?
                (default :obj:`False`, the final mini-batch will always
                 be visited even if it has fewer data than ``batch_size``)
            batch_count (int or None): The number of mini-batches to obtain
                in an epoch.  (default :obj:`None`, infinite mini-batches)

        Returns:
            tfsnippet.dataflow.DataFlow: A dataflow, with each mini-batch
                having numpy arrays ``([filename,] content, [meta-data...])``,
                according to the arguments.

        Raises:
            UnsupportedOperation: If ``RANDOM_SAMPLE`` capacity is absent.
        """
        if not self.capacity.can_random_sample():
            raise UnsupportedOperation()
        from .dataflow import DataFSRandomFlow
        return DataFSRandomFlow(
            fs=self.clone(),
            with_names=with_names,
            meta_keys=meta_keys,
            batch_size=batch_size,
            batch_count=batch_count,
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
            UnsupportedOperation: If ``RANDOM_SAMPLE`` capacity is absent,
                or ``meta_keys`` is specified, but ``READ_META`` capacity
                is absent.
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
            DataFileNotExist: If `filename` does not exist.
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
            DataFileNotExist: If `filename` does not exist.
        """
        with maybe_close(self.open(filename, 'r')) as f:
            return f.read()

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
            with maybe_close(self.open(filename, 'w')) as f:
                f.write(data)
        elif hasattr(data, 'read'):
            with maybe_close(self.open(filename, 'w')) as f:
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
            InvalidOpenMode: If the specified mode is not supported,
                e.g., ``mode == 'w'`` but ``WRITE_DATA`` capacity is absent.
            DataFileNotExist: If ``mode == 'r'`` but `filename` does not exist.
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
            tuple[str]: The keys of the meta data of the file.

        Raises:
            DataFileNotExist: If `filename` does not exist.
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
            tuple[any]: The meta values, corresponding to ``meta_keys``.
                If a requested key is absent for a file, :obj:`None` will
                take the place.

        Raises:
            DataFileNotExist: If `filename` does not exist.
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
        meta_keys = tuple(meta_keys or ())
        ret = []
        for name in filenames:
            try:
                ret.append(self.get_meta(name, meta_keys))
            except DataFileNotExist:
                ret.append(None)
        return ret

    def get_meta_dict(self, filename):
        """
        Get all the meta data of a file, as a dict.

        Args:
            filename (str): The name of the file.

        Returns:
            dict[str, any]: The meta values, as a dict.

        Raises:
            DataFileNotExist: If `filename` does not exist.
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
            DataFileNotExist: If `filename` does not exist.
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
            DataFileNotExist: If `filename` does not exist.
            UnsupportedOperation: If the ``WRITE_META`` capacity (and
                possibly the ``LIST_META`` capacity) is(are) absent.
        """
        self.clear_meta(filename)
        self.put_meta(filename, meta_dict, **meta_dict_kwargs)

    def clear_meta(self, filename):
        """
        Clear all the meta data of a file.

        Args:
            filename (str): The name of the file.

        Raises:
            DataFileNotExist: If `filename` does not exist.
            UnsupportedOperation: If the ``WRITE_META`` capacity (and
                possibly the ``LIST_META`` capacity) is(are) absent.
        """
        raise NotImplementedError()
