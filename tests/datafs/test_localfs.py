import unittest
from contextlib import contextmanager

import os

import six

from mltoolkit.utils import TemporaryDirectory, makedirs
from mltoolkit.datafs import *

from .standard_checks import StandardFSChecks, LocalFS


class LocalFSTestCase(unittest.TestCase, StandardFSChecks):

    def get_snapshot(self, fs):
        ret = {}
        for name in iter_files(fs.root_dir):
            with open(os.path.join(fs.root_dir, name), 'rb') as f:
                cnt = f.read()
            ret[name] = (cnt,)
        return ret

    @contextmanager
    def temporary_fs(self, snapshot=None):
        with TemporaryDirectory() as tempdir:
            if snapshot:
                for filename, (content,) in six.iteritems(snapshot):
                    file_path = os.path.join(tempdir, filename)
                    file_dir = os.path.split(file_path)[0]
                    makedirs(file_dir, exist_ok=True)
                    with open(file_path, 'wb') as f:
                        f.write(content)
            yield LocalFS(tempdir)

    def test_standard(self):
        self.run_standard_checks(DataFSCapacity.READ_WRITE_DATA)


if __name__ == '__main__':
    unittest.main()
