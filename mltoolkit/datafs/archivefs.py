import os
import tarfile
import zipfile

from .utils import ActiveFiles
from .base import *

__all__ = ['TarArchiveFS', 'ZipArchiveFS']


class _ArchiveFS(DataFS):
    """Base class for archive file based :class:`DataFS`."""

    def __init__(self, archive_file, strict):
        super(_ArchiveFS, self).__init__(
            capacity=DataFSCapacity(DataFSCapacity.READ_DATA),
            strict=strict
        )

        archive_file = os.path.abspath(archive_file)
        if not os.path.isfile(archive_file):
            raise IOError('Not a file: {!r}'.format(archive_file))
        self._archive_file = archive_file

    @property
    def archive_file(self):
        """Get the absolute path of the archive file."""
        return self._archive_file

    def clone(self):
        return self.__class__(self.archive_file, strict=self.strict)

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
    """Tar archive file based :class:`DataFS`."""

    def __init__(self, archive_file, strict=False):
        """
        Construct a new :class:`TarArchiveFS`.

        Args:
            archive_file (str): Path of the archive file.
            strict (bool): Whether or not this :class:`DataFS` works in
                strict mode?  (default :obj:`False`)
        """
        super(TarArchiveFS, self).__init__(archive_file, strict=strict)
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
                with self._file_obj.extractfile(mi) as f:
                    cnt = f.read()
                yield self._canonical_path(mi.name), cnt

    def open(self, filename, mode):
        self.init()
        if mode == 'r':
            mi = self._file_obj.getmember(filename)
            return self._active_files.add(self._file_obj.extractfile(mi))
        else:
            raise InvalidOpenMode(mode)

    def isfile(self, filename):
        try:
            mi = self._file_obj.getmember(filename)
            return not mi.isdir()
        except KeyError:
            return False


class ZipArchiveFS(_ArchiveFS):
    """Zip archive file based :class:`DataFS`."""

    def __init__(self, archive_file, strict=False):
        """
        Construct a new :class:`ZipArchiveFS`.

        Args:
            archive_file (str): Path of the archive file.
            strict (bool): Whether or not this :class:`DataFS` works in
                strict mode?  (default :obj:`False`)
        """
        super(ZipArchiveFS, self).__init__(archive_file, strict=strict)
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
                with self._file_obj.open(mi) as f:
                    cnt = f.read()
                yield self._canonical_path(mi.filename), cnt

    def open(self, filename, mode):
        self.init()
        if mode != 'r':
            raise InvalidOpenMode(mode)
        return self._active_files.add(self._file_obj.open(filename, mode=mode))

    def isfile(self, filename):
        try:
            mi = self._file_obj.getinfo(filename)
            return not self._isdir(mi)
        except KeyError:
            return False
