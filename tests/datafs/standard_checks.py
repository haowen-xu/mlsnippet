from contextlib import contextmanager

import pytest
import six

from mltoolkit.datafs import *


class StandardFSChecks(object):
    """
    Standard tests for :class:`DataFS` subclasses.

    To actually implement tests for a particular :class:`DataFS` subclass,
    inherit both the :class:`unittest.TestCase` and this class, and provide
    necessary utilities to support the checks, for example::

        class LocalFSTestCase(unittest.TestCase, StandardFSChecks):

            def get_snapshot(self, fs):
                # implement the method to take snapshot from a local `fs`

            @contextmanager
            def temporary_fs(self, snapshot=None):
                with TemporaryDirectory() as tempdir:
                    # populate the fs
                    yield LocalFS(tempdir)

            def test_standard(self):
                self.run_standard_checks(
                    DataFSCapacity.READ_WRITE_DATA)
    """

    def get_snapshot(self, fs):
        """
        Get the snapshot of the specified `fs`.

        Args:
            fs: The fs, where to collect the snapshot.

        Returns:
            The snapshot.
        """
        raise NotImplementedError()

    @contextmanager
    def temporary_fs(self, snapshot=None):
        """
        Initialize a temporary :class:`DataFS` with specified snapshot,
        and returns a context on this fs.

        Args:
            snapshot: The initial contents of the constructed fs.

        Yields:
            DataFS: The constructed fs.
        """
        raise NotImplementedError()

    def assert_snapshot_equals(self, expected, actual):
        """
        Assert the two snapshots are equal.

        Args:
            expected: The expected fs snapshot.
            actual: The actual fs snapshot.
        """
        self.assertDictEqual(expected, actual)

    def run_standard_checks(self, *capacity):
        """
        Run all the standard :class:`DataFS` tests.

        Args:
            *capacity: The expected capacity flags.
        """
        capacity = DataFSCapacity(*capacity)
        self.check_read(capacity)

    def check_read(self, capacity):
        names = ['a/1.txt', 'a/2.htm', 'b/1.md', 'b/2.rst', 'c']

        # first part: no meta data
        if six.PY2:
            to_bytes = lambda s: s
        else:
            to_bytes = lambda s: s.encode('utf-8')
        snapshot = {n: (to_bytes(n) + b' content',) for n in names}
        with self.temporary_fs(snapshot) as fs:
            # count, iter, list and sample names
            self.assertEquals(5, fs.count())
            self.assertListEqual(names, sorted(fs.iter_names()))
            self.assertIsInstance(fs.list_names(), list)
            self.assertListEqual(names, sorted(fs.list_names()))

            if capacity.can_random_sample():
                for repeated in range(10):
                    for k in range(len(names)):
                        samples = fs.sample_names(k)
                        self.assertIsInstance(samples, list)
                        self.assertEquals(k, len(samples))
                        for sample in samples:
                            self.assertIn(sample, names)
            else:
                with pytest.raises(UnsupportedOperation):
                    _ = fs.sample_names(1)

            # iter and sample files

