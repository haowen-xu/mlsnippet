import os
import tarfile
import unittest
import zipfile
from contextlib import contextmanager

import pytest
import six

from mltoolkit.datafs import *
from mltoolkit.utils import TemporaryDirectory, makedirs
from .standard_checks import StandardFSChecks


def canonical_path(name):
    return name.replace('\\', '/')


class TarArchiveFSTestCase(unittest.TestCase, StandardFSChecks):

    def get_snapshot(self, fs):
        ret = {}
        for mi in fs._file_obj:
            if not mi.isdir():
                with fs._file_obj.extractfile(mi) as f:
                    cnt = f.read()
                ret[canonical_path(mi.name)] = (cnt,)
        return ret

    @contextmanager
    def temporary_fs(self, snapshot=None, **kwargs):
        with TemporaryDirectory() as tempdir:
            archive_file = os.path.join(tempdir, 'archive.tar.gz')
            tempdir = os.path.join(tempdir, 'temp')
            if snapshot:
                for filename, payload in six.iteritems(snapshot):
                    content = payload[0]
                    file_path = os.path.join(tempdir, filename)
                    file_dir = os.path.split(file_path)[0]
                    makedirs(file_dir, exist_ok=True)
                    with open(file_path, 'wb') as f:
                        f.write(content)
            with tarfile.open(archive_file, 'w:gz') as fobj:
                if snapshot:
                    for filename in os.listdir(tempdir):
                        temp_path = os.path.join(tempdir, filename)
                        fobj.add(temp_path, arcname=filename)
            with TarArchiveFS(archive_file, **kwargs) as fs:
                yield fs

    def test_standard(self):
        self.run_standard_checks(DataFSCapacity.READ_DATA)

    def test_errors(self):
        with pytest.raises(IOError, match='Not a file'):
            _ = TarArchiveFS('/this/path/cannot/be/a/file')
        with TemporaryDirectory() as tempdir:
            with pytest.raises(IOError, match='Not a file'):
                _ = TarArchiveFS(tempdir)


class ZipArchiveFSTestCase(unittest.TestCase, StandardFSChecks):

    def get_snapshot(self, fs):
        ret = {}
        for mi in fs._file_obj.infolist():
            if mi.filename[-1] != '/':
                with fs._file_obj.open(mi) as f:
                    cnt = f.read()
                ret[canonical_path(mi.filename)] = (cnt,)
        return ret

    @contextmanager
    def temporary_fs(self, snapshot=None, **kwargs):
        with TemporaryDirectory() as tempdir:
            archive_file = os.path.join(tempdir, 'archive.zip')
            tempdir = os.path.join(tempdir, 'temp')
            if snapshot:
                for filename, payload in six.iteritems(snapshot):
                    content = payload[0]
                    file_path = os.path.join(tempdir, filename)
                    file_dir = os.path.split(file_path)[0]
                    makedirs(file_dir, exist_ok=True)
                    with open(file_path, 'wb') as f:
                        f.write(content)
            with zipfile.ZipFile(archive_file, 'w') as fobj:
                if snapshot:
                    def g(arcname, path):
                        for filename in os.listdir(path):
                            file_arcname = (arcname + '/' + filename
                                            if arcname else filename)
                            file_path = os.path.join(path, filename)
                            fobj.write(file_path, arcname=file_arcname)
                            if os.path.isdir(file_path):
                                g(file_arcname, file_path)
                    g('', tempdir)
            with ZipArchiveFS(archive_file, **kwargs) as fs:
                yield fs

    def test_standard(self):
        self.run_standard_checks(DataFSCapacity.READ_DATA)

    def test_errors(self):
        with pytest.raises(IOError, match='Not a file'):
            _ = ZipArchiveFS('/this/path/cannot/be/a/file')
        with TemporaryDirectory() as tempdir:
            with pytest.raises(IOError, match='Not a file'):
                _ = ZipArchiveFS(tempdir)


if __name__ == '__main__':
    unittest.main()
