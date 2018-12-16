import numpy as np
import six
from numpy.random import RandomState

from tfsnippet.dataflow import DataFlow
from tfsnippet.utils import AutoInitAndCloseable, minibatch_slices_iterator

from .base import DataFS

__all__ = [
    'DataFSForwardFlow',
    'DataFSIndexedFlow',
    'DataFSRandomFlow'
]


class _BaseDataFSFlow(DataFlow, AutoInitAndCloseable):
    """
    Base class for all :class:`DataFS` derived :class:`DataFlow` subclasses.
    """

    def __init__(self, fs, batch_size, with_names=True, meta_keys=None,
                 skip_incomplete=False):
        """
        Initialize all internal states of the :class:`_BaseDataFSFlow`.

        Args:
            fs (DataFS): The data fs instance, where to read data.
            batch_size (int): Size of each mini-batch.
            with_names (bool): Whether or not to include the file names
                in mini-batches? (default :obj:`True`)
            meta_keys (None or Iterable[str]): The keys of the meta data
                to be included in mini-batches. (default :obj:`None`)
            skip_incomplete (bool): Whether or not to exclude a mini-batch,
                if it has fewer data than ``batch_size``?
                (default :obj:`False`, the final mini-batch will always
                 be visited even if it has fewer data than ``batch_size``)
        """
        super(_BaseDataFSFlow, self).__init__()
        self._fs = fs  # type: DataFS
        self._batch_size = batch_size
        self._with_names = with_names
        self._meta_keys = tuple(meta_keys) if meta_keys is not None else None
        self._skip_incomplete = skip_incomplete

    @property
    def batch_size(self):
        """
        Get the size of each mini-batch.

        Returns:
            int: The size of each mini-batch.
        """
        return self._batch_size

    @property
    def skip_incomplete(self):
        """
        Whether or not to exclude a mini-batch, if it has fewer data than
        ``batch_size``?
        """
        return self._skip_incomplete

    @property
    def fs(self):
        """
        Get the data fs instance.

        Returns:
            DataFS: The data fs instance.
        """
        return self._fs

    @property
    def with_names(self):
        """
        Whether or not to include the file names in mini-batches?
        """
        return self._with_names

    @property
    def meta_keys(self):
        """
        Get the keys of the meta data to be included in mini-batches.

        Returns:
            None or tuple[str]: The meta keys to retrieve, or None if not
                configured.
        """
        return self._meta_keys

    def _init(self):
        self.fs.init()

    def _close(self):
        self.fs.close()


class _BatchArrayGenerator(object):
    """
    A helper class for gathering data from :class:`DataFS` into mini-batches.
    """

    def __init__(self, batch_size, with_names, meta_keys):
        meta_keys = meta_keys or ()
        self.batch_size = batch_size
        self.with_names = with_names
        self.meta_keys = meta_keys
        self.buffers = [
            [] for _ in range(int(with_names) +  # optional file name
                              1 +  # file data
                              len(meta_keys))  # optional file meta
        ]

        if with_names:
            def add(name, data, meta=()):
                self.buffers[0].append(name)
                self.buffers[1].append(data)
                for buf, val in zip(self.buffers[2:], meta):
                    buf.append(val)

        else:
            def add(name, data, meta=()):
                self.buffers[0].append(data)
                for buf, val in zip(self.buffers[1:], meta):
                    buf.append(val)
        self.add = add

    def to_arrays(self):
        if self.with_names:
            return ((np.asarray(self.buffers[0], dtype=str),
                     np.asarray(self.buffers[1], dtype=six.binary_type),) +
                    tuple(np.asarray(buf) for buf in self.buffers[2:]))
        else:
            return ((np.asarray(self.buffers[0], dtype=six.binary_type),) +
                    tuple(np.asarray(buf) for buf in self.buffers[1:]))

    def clear_all(self):
        for buf in self.buffers:
            del buf[:]

    @property
    def full_batch(self):
        return len(self.buffers[0]) >= self.batch_size

    @property
    def not_empty(self):
        return len(self.buffers[0]) >= 0


class DataFSForwardFlow(_BaseDataFSFlow):
    """
    A :class:`DataFS` derived :class:`DataFlow`, iterating through mini-batches
    in a forward-only fashion (data are obtained by :meth:`iter_files`).
    """

    def __init__(self, fs, batch_size, with_names=True, meta_keys=None,
                 skip_incomplete=False):
        """
        Construct a new :class:`DataFSForwardFlow`.

        Args:
            fs (DataFS): The data fs instance, where to read data.
            batch_size (int): Size of each mini-batch.
            with_names (bool): Whether or not to include the file names
                in mini-batches? (default :obj:`True`)
            meta_keys (None or Iterable[str]): The keys of the meta data
                to be included in mini-batches. (default :obj:`None`)
            skip_incomplete (bool): Whether or not to exclude a mini-batch,
                if it has fewer data than ``batch_size``? (default
                :obj:`False`, the final mini-batch will always be visited even
                if it has fewer data than ``batch_size``)
        """
        super(DataFSForwardFlow, self).__init__(
            fs=fs,
            batch_size=batch_size,
            with_names=with_names,
            meta_keys=meta_keys,
            skip_incomplete=skip_incomplete
        )

    def _minibatch_iterator(self):
        g = _BatchArrayGenerator(
            self.batch_size, self.with_names, self.meta_keys)
        for f in self.fs.iter_files(meta_keys=self.meta_keys):
            g.add(f[0], f[1], f[2:])
            if g.full_batch:
                yield g.to_arrays()
                g.clear_all()
        if g.not_empty and (g.full_batch or not self.skip_incomplete):
            yield g.to_arrays()


class DataFSIndexedFlow(_BaseDataFSFlow):
    """
    A :class:`DataFS` derived :class:`DataFlow`, iterating through mini-batches
    according to given names (data are obtaining by :meth:`retrieve`).
    """

    def __init__(self, fs, batch_size, names, with_names=True, meta_keys=None,
                 shuffle=False, skip_incomplete=False, random_state=None):
        """
        Construct a new :class:`DataFSIndexedFlow`.

        Args:
            fs (DataFS): The data fs instance, where to read data.
            batch_size (int): Size of each mini-batch.
            names (list[str] or np.ndarray[str]): The names to retrieve.
            with_names (bool): Whether or not to include the file names
                in mini-batches? (default :obj:`True`)
            meta_keys (None or Iterable[str]): The keys of the meta data
                to be included in mini-batches. (default :obj:`None`)
            shuffle (bool): Whether or not to shuffle the name indices
                before each epoch?  (default :obj:`False`)
            skip_incomplete (bool): Whether or not to exclude a mini-batch,
                if it has fewer data than ``batch_size``? (default
                :obj:`False`, the final mini-batch will always be visited even
                if it has fewer data than ``batch_size``)
            random_state (RandomState): Optional numpy RandomState for
                shuffling data before each epoch.  (default :obj:`None`,
                use the global :class:`RandomState`).
        """
        super(DataFSIndexedFlow, self).__init__(
            fs=fs,
            batch_size=batch_size,
            with_names=with_names,
            meta_keys=meta_keys,
            skip_incomplete=skip_incomplete
        )
        self._names = np.asarray(names, dtype=str)
        self._is_shuffled = shuffle
        self._cached_indices = None  # np.ndarray
        self._random_state = random_state or np.random

    @property
    def names(self):
        """
        Get the names of files to retrieve.

        Returns:
            np.ndarray[str]: The names, as numpy array.
        """
        return self._names

    @property
    def is_shuffled(self):
        """
        Whether or not to shuffle the names before each epoch?
        """
        return self._is_shuffled

    def _shuffled_indices_iterator(self):
        # reuse indices
        if self._cached_indices is None:
            indices_dtype = (
                np.int32 if len(self.names) <= np.iinfo(np.int32).max
                else np.int64)
            self._cached_indices = np.arange(
                len(self.names), dtype=indices_dtype)
        indices = self._cached_indices

        # shuffle indices
        self._random_state.shuffle(indices)

        for s in minibatch_slices_iterator(
                length=len(self.names),
                batch_size=self.batch_size,
                skip_incomplete=self.skip_incomplete):
            yield indices[s]

    def _normal_indices_iterator(self):
        return minibatch_slices_iterator(
            length=len(self.names),
            batch_size=self.batch_size,
            skip_incomplete=self.skip_incomplete
        )

    def _minibatch_iterator(self):
        # iterate through mini-batches
        if self.is_shuffled:
            indices_iter = self._shuffled_indices_iterator()
        else:
            indices_iter = self._normal_indices_iterator()

        # for gathering batch arrays
        g = _BatchArrayGenerator(
            self.batch_size, self.with_names, self.meta_keys)

        # produce the mini-batches
        meta_keys = tuple(self.meta_keys or ())
        for s in indices_iter:
            s_names = self.names[s]
            s_data = [self.fs.retrieve(name, meta_keys=meta_keys)
                      for name in s_names]
            for n, d in zip(s_names, s_data):
                g.add(n, d[0], d[1:])
            yield g.to_arrays()
            g.clear_all()


class DataFSRandomFlow(_BaseDataFSFlow):
    """
    A :class:`DataFS` derived :class:`DataFlow`, obtaining random samples
    from the :class:`DataFS`.
    """

    def __init__(self, fs, batch_size, with_names=True, meta_keys=None,
                 batch_count=None, skip_incomplete=False):
        """
        Construct a new :class:`DataFSRandomFlow`.

        Args:
            fs (DataFS): The data fs instance, where to read data.
            batch_size (int): Size of each mini-batch.
            with_names (bool): Whether or not to include the file names
                in mini-batches? (default :obj:`True`)
            meta_keys (None or Iterable[str]): The keys of the meta data
                to be included in mini-batches. (default :obj:`None`)
            batch_count (int or None): The number of mini-batches to obtain
                in an epoch.  (default :obj:`None`, infinite mini-batches)
            skip_incomplete (bool): Whether or not to exclude a mini-batch,
                if it has fewer data than ``batch_size``? (default
                :obj:`False`, the final mini-batch will always be visited even
                if it has fewer data than ``batch_size``)
        """
        super(DataFSRandomFlow, self).__init__(
            fs, batch_size=batch_size, with_names=with_names,
            meta_keys=meta_keys, skip_incomplete=skip_incomplete
        )
        if batch_count is not None:
            if batch_count <= 0:
                raise ValueError('`batch_count` must be positive.')
        self._batch_count = batch_count

        # the loop generator
        if batch_count is None:
            def loop_generator():
                while True:
                    yield
        else:
            if six.PY2:
                def loop_generator():
                    return xrange(batch_count)
            else:
                def loop_generator():
                    return range(batch_count)
        self._loop_generator = loop_generator

    @property
    def batch_count(self):
        """Get the number of mini-batches to obtain in an epoch."""
        return self._batch_count

    def _minibatch_iterator(self):
        g = _BatchArrayGenerator(batch_size=self.batch_size,
                                 with_names=self.with_names,
                                 meta_keys=self.meta_keys)
        for _ in self._loop_generator():
            batch = self.fs.sample_files(self.batch_size, self.meta_keys)
            if batch:
                for b in batch:
                    g.add(b[0], b[1], b[2:])
                if g.full_batch or not self.skip_incomplete:
                    yield g.to_arrays()
                g.clear_all()
