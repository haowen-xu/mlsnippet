import functools
import re

import numpy as np
import six

from tfsnippet.dataflow import DataFlow
from tfsnippet.utils import AutoInitAndCloseable, minibatch_slices_iterator

from .base import DataFS

__all__ = ['DataFSFlow', 'DataFSRandomFlow']


class _BaseDataFSFlow(DataFlow, AutoInitAndCloseable):

    def __init__(self, fs, batch_size, with_names=False, meta_keys=None,
                 skip_incomplete=False):
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
        Whether or not to exclude the last mini-batch if it is incomplete?
        """
        return self._skip_incomplete

    @property
    def fs(self):
        return self._fs

    @property
    def with_names(self):
        return self._with_names

    @property
    def meta_keys(self):
        return self._meta_keys

    def _init(self):
        self.fs.init()

    def _close(self):
        self.fs.close()


class _BatchArrayGenerator(object):

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


class DataFSFlow(_BaseDataFSFlow):

    def __init__(self, fs, batch_size, with_names=False, meta_keys=None,
                 shuffle=False, skip_incomplete=False, names_pattern=None):
        super(DataFSFlow, self).__init__(
            fs, batch_size=batch_size, with_names=with_names,
            meta_keys=meta_keys, skip_incomplete=skip_incomplete
        )
        self._is_shuffled = shuffle
        self._cached_filenames = None
        self._cached_indices = None
        self._names_pattern = re.compile(names_pattern) \
            if names_pattern is not None else None

        # derive the function to test file name
        if self._names_pattern is not None:
            def _is_name_matched(n):
                return self._names_pattern.match(n) is not None
        else:
            def _is_name_matched(n):
                return True
        self._is_name_matched = _is_name_matched

    @property
    def is_shuffled(self):
        """
        Whether or not the data are first shuffled before iterated through
        mini-batches?
        """
        return self._is_shuffled

    @property
    def names_pattern(self):
        """
        Get the regex pattern for filtering the file names.

        Returns:
            None or regex: The regex pattern for file names, or :obj:`None`
                if the pattern is not set.
        """
        return self._names_pattern

    def __natural_iterator(self):
        g = _BatchArrayGenerator(
            self.batch_size, self.with_names, self.meta_keys)
        for f in self.fs.iter_files(meta_keys=self.meta_keys):
            if self._is_name_matched(f[0]):
                g.add(f[0], f[1], f[2:])
                if g.full_batch:
                    yield g.to_arrays()
                    g.clear_all()
        if g.not_empty and (g.full_batch or not self.skip_incomplete):
            yield g.to_arrays()

    def __shuffle_iterator(self):
        # cache filenames
        if self._cached_filenames is None:
            if self._names_pattern is None:
                self._cached_filenames = np.asarray(self.fs.list_names())
            else:
                self._cached_filenames = np.asarray([
                    n for n in self.fs.iter_names()
                    if self._is_name_matched(n)
                ])
        names = self._cached_filenames

        # reuse indices
        if self._cached_indices is None:
            indices_dtype = (np.int32 if len(names) <= np.iinfo(np.int32).max
                             else np.int64)
            self._cached_indices = np.arange(len(names), dtype=indices_dtype)
        indices = self._cached_indices

        # shuffle indices
        np.random.shuffle(indices)

        # iterate through mini-batches
        g = _BatchArrayGenerator(
            self.batch_size, self.with_names, self.meta_keys)
        mkiter = functools.partial(minibatch_slices_iterator,
                                   length=len(names),
                                   batch_size=self.batch_size,
                                   skip_incomplete=self.skip_incomplete)

        if self.meta_keys:
            for s in mkiter():
                s_names = names[indices[s]]
                s_data = [self.fs.retrieve(name) for name in s_names]
                s_meta = self.fs.batch_get_meta(s_names, self.meta_keys)
                for n, d, m in zip(s_names, s_data, s_meta):
                    g.add(n, d, m)
                yield g.to_arrays()
                g.clear_all()
        else:
            for s in mkiter():
                s_names = names[indices[s]]
                s_data = [self.fs.retrieve(name) for name in s_names]
                for n, d in zip(s_names, s_data):
                    g.add(n, d)
                yield g.to_arrays()
                g.clear_all()

    def _minibatch_iterator(self):
        if self.is_shuffled:
            return self.__shuffle_iterator()
        else:
            return self.__natural_iterator()


class DataFSRandomFlow(_BaseDataFSFlow):

    def __init__(self, fs, batch_size, with_names=False, meta_keys=None,
                 epoch_size=None, skip_incomplete=False):
        super(DataFSRandomFlow, self).__init__(
            fs, batch_size=batch_size, with_names=with_names,
            meta_keys=meta_keys, skip_incomplete=skip_incomplete
        )
        if epoch_size is not None:
            if epoch_size % self.batch_size != 0:
                raise ValueError('`epoch_size` must be multiples of '
                                 '`batch_size`.')
            if epoch_size <= 0:
                raise ValueError('`epoch_size` must be positive.')
        self._epoch_size = epoch_size

        # the loop generator
        if epoch_size is None:
            def loop_generator():
                while True:
                    yield
        else:
            if six.PY2:
                def loop_generator():
                    return xrange(epoch_size / self.batch_size)
            else:
                def loop_generator():
                    return range(epoch_size / self.batch_size)
        self._loop_generator = loop_generator

        # the batch array generator
        if self.with_names:
            def make_batch_arrays(batch):
                return ((np.asarray(batch[0], dtype=str),
                         np.asarray(batch[1], dtype=six.binary_type),) +
                        tuple(np.asarray(buf) for buf in batch[2:]))
        else:
            def make_batch_arrays(batch):
                return ((np.asarray(batch[1], dtype=six.binary_type),) +
                        tuple(np.asarray(buf) for buf in batch[2:]))
        self._make_batch_arrays = make_batch_arrays

    @property
    def epoch_size(self):
        return self._epoch_size

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
