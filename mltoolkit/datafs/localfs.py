import os

from .utils import ActiveFiles
from .base import *

__all__ = ['LocalFS']


def iter_files(root_dir, sep='/'):
    def f(parent_path, parent_name):
        for f_name in os.listdir(parent_path):
            f_child_path = parent_path + os.sep + f_name
            f_child_name = parent_name + sep + f_name
            if os.path.isdir(f_child_path):
                for s in f(f_child_path, f_child_name):
                    yield s
            else:
                yield f_child_name

    for name in os.listdir(root_dir):
        child_path = root_dir + os.sep + name
        if os.path.isdir(child_path):
            for x in f(child_path, name):
                yield x
        else:
            yield name


class LocalFS(DataFS):

    def __init__(self, root_dir):
        root_dir = os.path.abspath(root_dir)
        if not os.path.isdir(root_dir):
            raise IOError('Not a directory: {!r}'.format(root_dir))
        self._root_dir = root_dir
        self._capacity = DataFSCapacity(
            DataFSCapacity.READ_DATA,
            DataFSCapacity.WRITE_DATA,
        )
        self._active_files = ActiveFiles()

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def capacity(self):
        return self._capacity

    def clone(self):
        return LocalFS(self.root_dir)

    def _init(self):
        pass

    def _close(self):
        self._active_files.close_all()

    def iter_names(self):
        self.init()
        return iter_files(self.root_dir)

    def sample_names(self, n_samples):
        raise UnsupportedOperation()

    def open(self, filename, mode):
        self.init()
        file_path = os.path.join(self.root_dir, filename)
        if mode == 'r':
            return self._active_files.add(open(file_path, 'rb'))
        elif mode == 'w':
            return self._active_files.add(open(file_path, 'wb'))
        else:
            raise ValueError('Unsupported open mode {!r}'.format(mode))

    def list_meta(self, filename):
        raise UnsupportedOperation()

    def get_meta(self, filename, meta_keys):
        raise UnsupportedOperation()

    def put_meta(self, filename, meta_dict=None, **meta_dict_kwargs):
        raise UnsupportedOperation()

    def clear_meta(self, filename):
        raise UnsupportedOperation()
