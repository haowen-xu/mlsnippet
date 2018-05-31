from contextlib import contextmanager
from io import BytesIO

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
            fs: The fs, from where to collect the snapshot.

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
        if capacity.can_write_data():
            self.check_write(capacity)

    def check_read(self, capacity):
        names = ['a/1.txt', 'a/2.htm', 'b/1.md', 'b/2.rst', 'c']

        if six.PY2:
            to_bytes = lambda s: s
        else:
            to_bytes = lambda s: s.encode('utf-8')
        get_content = lambda n: to_bytes(n) + b' content'
        snapshot = {n: (get_content(n),) for n in names}
        with self.temporary_fs(snapshot) as fs:
            self.assertDictEqual(snapshot, self.get_snapshot(fs))

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
            self.assertListEqual(
                [(n, to_bytes(n) + b' content') for n in names],
                list(fs.iter_files())
            )

            if capacity.can_random_sample():
                for repeated in range(10):
                    for k in range(len(names)):
                        samples = fs.sample_files(k)
                        self.assertIsInstance(samples, list)
                        self.assertEquals(k, len(samples))
                        for sample in samples:
                            self.assertIn(sample[0], names)
                            self.assertEquals(get_content(sample[0]), sample[1])
            else:
                with pytest.raises(UnsupportedOperation):
                    _ = fs.sample_names(1)

            # retrieve, get data and open
            for n in names:
                self.assertEquals(get_content(n), fs.retrieve(n))
                self.assertEquals(get_content(n), fs.get_data(n))
                with fs.open(n, 'r') as f:
                    self.assertEquals(get_content(n), f.read())

            # isfile, batch_isfile
            for n in names:
                self.assertTrue(fs.isfile(n))
                self.assertFalse(fs.isfile(n + '.invalid-ext'))
            self.assertListEqual([True] * len(names), fs.batch_isfile(names))
            self.assertListEqual(
                [True, False] * len(names),
                fs.batch_isfile(sum(
                    [[n, n + '.invalid-ext'] for n in names], []))
            )

    def check_write(self, capacity):
        with self.temporary_fs() as fs:
            self.assertDictEqual({}, self.get_snapshot(fs))

            # put_data
            fs.put_data('a/1.txt', b'a/1.txt content')
            fs.put_data('b/2.txt', BytesIO(b'b/2.txt content'))
            with pytest.raises(TypeError, match='`data` must be bytes or a '
                                                'file-like object'):
                fs.put_data('err1.txt', u'')
            with pytest.raises(TypeError, match='`data` must be bytes or a '
                                                'file-like object'):
                fs.put_data('err2.txt', object())

            # open
            with fs.open('c/3.txt', 'w') as f:
                f.write(b'c/3.txt content')

            self.assertDictEqual(
                {
                    'a/1.txt': (b'a/1.txt content',),
                    'b/2.txt': (b'b/2.txt content',),
                    'c/3.txt': (b'c/3.txt content',)
                },
                self.get_snapshot(fs)
            )
