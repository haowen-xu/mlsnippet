import functools

import numpy as np

from tfsnippet.dataflow import DataFlow
from tfsnippet.utils import InitDestroyable, minibatch_slices_iterator

from .base import DataFS

__all__ = ['DataFSFlow']


class _BaseDataFSFlow(DataFlow, InitDestroyable):

    def __init__(self, fs):
        self._fs = fs  # type: DataFS

    @property
    def fs(self):
        return self._fs

    def _init(self):
        self.fs.init()

    def _destroy(self):
        self.fs.destroy()


class DataFSFlow(_BaseDataFSFlow):

    def __init__(self, fs, batch_size, with_names=False, shuffle=False,
                 skip_incomplete=False):
        _BaseDataFSFlow.__init__(self, fs)
        InitDestroyable.__init__(self)
        self._batch_size = batch_size
        self._with_names = with_names
        self._is_shuffled = shuffle
        self._skip_incomplete = skip_incomplete
        self._cached_filenames = None
        self._cached_indices = None

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
    def is_shuffled(self):
        """
        Whether or not the data are first shuffled before iterated through
        mini-batches?
        """
        return self._is_shuffled

    @property
    def with_names(self):
        return self._with_names

    def __natural_iterator(self):
        if self.with_names:
            n_buf, d_buf = [], []
            for n, d in self.fs.iter_files():
                n_buf.append(n)
                d_buf.append(d)
                if len(d_buf) >= self.batch_size:
                    yield np.asarray(n_buf), np.asarray(d_buf)
                    n_buf.clear()
                    d_buf.clear()
            if d_buf and (len(d_buf) >= self.batch_size or
                          not self.skip_incomplete):
                yield np.asarray(n_buf), np.asarray(d_buf)
        else:
            d_buf = []
            for _, d in self.fs.iter_files():
                d_buf.append(d)
                if len(d_buf) >= self.batch_size:
                    yield np.asarray(d_buf),
                    d_buf.clear()
            if d_buf and (len(d_buf) >= self.batch_size or
                          not self.skip_incomplete):
                yield np.asarray(d_buf),

    def __shuffle_iterator(self):
        # cache filenames
        if self._cached_filenames is None:
            self._cached_filenames = np.asarray(self.fs.list_names())
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
        g = functools.partial(minibatch_slices_iterator,
                              length=len(names),
                              batch_size=self.batch_size,
                              skip_incomplete=self.skip_incomplete)
        if self.with_names:
            for s in g():
                s_names = names[indices[s]]
                s_data = np.asarray([self.fs.retrieve(n) for n in s_names])
                yield s_names, s_data
        else:
            for s in g():
                yield np.asarray(
                    [self.fs.retrieve(n) for n in names[indices[s]]])

    def _minibatch_iterator(self):
        if self.is_shuffled:
            return self.__shuffle_iterator()
        else:
            return self.__natural_iterator()
