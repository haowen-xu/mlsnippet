import os
import unittest
from contextlib import contextmanager

import pytest
import six

from mltoolkit.utils import TemporaryDirectory, makedirs, iter_files
from mltoolkit.datafs import *
from .standard_checks import StandardFSChecks


class LocalFSTestCase(unittest.TestCase, StandardFSChecks):

    def get_snapshot(self, fs):
        ret = {}
        for name in iter_files(fs.root_dir):
            with open(os.path.join(fs.root_dir, name), 'rb') as f:
                cnt = f.read()
            ret[name] = (cnt,)
        return ret

    @contextmanager
    def temporary_fs(self, snapshot=None, **kwargs):
        with TemporaryDirectory() as tempdir:
            if snapshot:
                for filename, payload in six.iteritems(snapshot):
                    content = payload[0]
                    file_path = os.path.join(tempdir, filename)
                    file_dir = os.path.split(file_path)[0]
                    makedirs(file_dir, exist_ok=True)
                    with open(file_path, 'wb') as f:
                        f.write(content)
            with LocalFS(tempdir, **kwargs) as fs:
                yield fs

    def test_standard(self):
        self.run_standard_checks(DataFSCapacity.READ_WRITE_DATA)

    def test_errors(self):
        with pytest.raises(IOError, match='Not a directory'):
            _ = LocalFS('/this/path/cannot/be/a/directory')
        with TemporaryDirectory() as tempdir:
            f_path = os.path.join(tempdir, '1.dat')
            with open(f_path, 'wb') as f:
                f.write(b'')
            with pytest.raises(IOError, match='Not a directory'):
                _ = LocalFS(f_path)


if __name__ == '__main__':
    unittest.main()
