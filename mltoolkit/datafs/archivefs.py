import os
import tarfile
import zipfile

from .utils import ActiveFiles
from .base import *

try:
    import rarfile
    rarfile.PATH_SEP = '/'
except ImportError:
    rarfile = None

__all__ = ['TarArchiveFS', 'ZipArchiveFS', 'RarArchiveFS']


class _ArchiveFS(DataFS):

    def __init__(self, archive_file, capacity):
        archive_file = os.path.abspath(archive_file)
        if not os.path.isfile(archive_file):
            raise IOError('Not a file: {!r}'.format(archive_file))
        self._archive_file = archive_file
        self._capacity = capacity

    @property
    def archive_file(self):
        return self._archive_file

    @property
    def capacity(self):
        return self._capacity

    def clone(self):
        return self.__class__(self.archive_file)

    def _canonical_path(self, path):
        return path.replace('\\', '/')

    def sample_names(self, n_samples):
        raise UnsupportedOperation()

    def list_meta(self, filename):
        raise UnsupportedOperation()

    def get_meta(self, filename, meta_keys):
        raise UnsupportedOperation()

    def put_meta(self, filename, meta_dict=None, **meta_dict_kwargs):
        raise UnsupportedOperation()

    def clear_meta(self, filename):
        raise UnsupportedOperation()


class TarArchiveFS(_ArchiveFS):

    def __init__(self, archive_file):
        super(TarArchiveFS, self).__init__(
            archive_file, DataFSCapacity(DataFSCapacity.READ_DATA))
        self._file_obj = None  # type: tarfile.TarFile
        self._active_files = ActiveFiles()

    def _init(self):
        self._file_obj = tarfile.open(self.archive_file, 'r')

    def _close(self):
        self._active_files.close_all()
        self._file_obj.close()

    def iter_names(self):
        self.init()
        for mi in self._file_obj:
            if not mi.isdir():
                yield self._canonical_path(mi.name)

    def iter_files(self, meta_keys=None):
        if meta_keys:
            raise UnsupportedOperation()
        self.init()
        for mi in self._file_obj:
            if not mi.isdir():
                yield (self._canonical_path(mi.name),
                       self._file_obj.extractfile(mi))

    def open(self, filename, mode):
        self.init()
        if mode == 'r':
            mi = self._file_obj.getmember(filename)
            return self._active_files.add(self._file_obj.extractfile(mi))
        else:
            raise ValueError('Unsupported open mode {!r}'.format(mode))


class ZipArchiveFS(_ArchiveFS):

    def __init__(self, archive_file):
        super(ZipArchiveFS, self).__init__(
            archive_file,
            DataFSCapacity(DataFSCapacity.READ_DATA)
        )
        self._file_obj = None  # type: zipfile.ZipFile
        self._active_files = ActiveFiles()

    def _init(self):
        self._file_obj = zipfile.ZipFile(self.archive_file, 'r')

    def _close(self):
        self._active_files.close_all()
        self._file_obj.close()

    def _isdir(self, member_info):
        return member_info.filename[-1] == '/'

    def iter_names(self):
        self.init()
        for mi in self._file_obj.infolist():
            if not self._isdir(mi):
                yield self._canonical_path(mi.filename)

    def iter_files(self, meta_keys=None):
        if meta_keys:
            raise UnsupportedOperation()
        self.init()
        for mi in self._file_obj.infolist():
            if not self._isdir(mi):
                yield (self._canonical_path(mi.filename),
                       self._file_obj.open(mi))

    def open(self, filename, mode):
        self.init()
        if mode != 'r':
            raise ValueError('Unsupported open mode {!r}'.format(mode))
        return self._active_files.add(self._file_obj.open(filename, mode=mode))


class RarArchiveFS(_ArchiveFS):

    def __init__(self, archive_file):
        if rarfile is None:
            raise RuntimeError('Python package `rarfile` is missing.')
        super(RarArchiveFS, self).__init__(
            archive_file, DataFSCapacity(DataFSCapacity.READ_DATA))
        self._file_obj = None  # type: rarfile.RarFile

    def _init(self):
        self._file_obj = rarfile.RarFile(self.archive_file, 'r')

    def _close(self):
        self._active_files.close_all()
        self._file_obj.close()

    def iter_names(self):
        self.init()
        for mi in self._file_obj.infolist():
            if not mi.isdir():
                yield self._canonical_path(mi.filename)

    def iter_files(self, meta_keys=None):
        if meta_keys:
            raise UnsupportedOperation()
        self.init()
        for mi in self._file_obj.infolist():
            if not mi.isdir():
                yield (self._canonical_path(mi.filename),
                       self._file_obj.open(mi))

    def open(self, filename, mode):
        self.init()
        if mode == 'r':
            mi = self._file_obj.getinfo(filename)
            return self._active_files.add(self._file_obj.open(mi))
        else:
            raise ValueError('Unsupported open mode {!r}'.format(mode))
