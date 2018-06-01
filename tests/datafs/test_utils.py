import functools
import gc
import os
import unittest

import pytest
from mock import Mock

from mltoolkit.datafs import ActiveFiles, iter_files
from mltoolkit.utils import makedirs, TemporaryDirectory


class ActiveFilesTestCase(unittest.TestCase):

    def test_close_all(self):
        def raises_error(err_cls, *args, **kwargs):
            raise err_cls(*args, **kwargs)

        f1 = Mock(close=Mock(return_value=None))
        f2 = Mock(close=Mock(
            wraps=functools.partial(raises_error, TypeError, 'f2')))
        f3 = Mock(close=Mock(
            wraps=functools.partial(raises_error, ValueError, 'f3')))

        active_files = ActiveFiles()
        self.assertIs(f1, active_files.add(f1))
        self.assertIs(f2, active_files.add(f2))
        self.assertIs(f3, active_files.add(f3))
        self.assertFalse(f1.close.called)
        self.assertFalse(f2.close.called)
        self.assertFalse(f3.close.called)
        active_files.close_all()
        self.assertTrue(f1.close.called)
        self.assertTrue(f2.close.called)
        self.assertTrue(f3.close.called)

    def test_weak_ref(self):
        def set_marker():
            marker[0] = 1
        marker = [0]
        active_files = ActiveFiles()

        def mkobj():
            f = Mock(close=Mock(wraps=set_marker))
            active_files.add(f)

        mkobj()
        gc.collect()
        active_files.close_all()
        self.assertEquals(0, marker[0])

    def test_reraise(self):
        def raises_error(err_cls, *args, **kwargs):
            raise err_cls(*args, **kwargs)

        # test keyboard interrupt
        active_files = ActiveFiles()
        f = active_files.add(
            Mock(close=Mock(
                wraps=functools.partial(raises_error, KeyboardInterrupt))))
        with pytest.raises(KeyboardInterrupt):
            active_files.close_all()
        self.assertTrue(f.close.called)

        # test system exit
        active_files = ActiveFiles()
        f = active_files.add(
            Mock(close=Mock(
                wraps=functools.partial(raises_error, SystemExit))))
        with pytest.raises(SystemExit):
            active_files.close_all()
        self.assertTrue(f.close.called)


class IterFilesTestCase(unittest.TestCase):

    def test_iter_files(self):
        names = ['a/1.txt', 'a/2.txt', 'a/b/1.txt', 'a/b/2.txt',
                 'b/1.txt', 'b/2.txt', 'c.txt']

        with TemporaryDirectory() as tempdir:
            for name in names:
                f_path = os.path.join(tempdir, name)
                f_dir = os.path.split(f_path)[0]
                makedirs(f_dir, exist_ok=True)
                with open(f_path, 'wb') as f:
                    f.write(b'')

            self.assertListEqual(names, sorted(iter_files(tempdir)))
            self.assertListEqual(names, sorted(iter_files(tempdir + '/a/../')))


if __name__ == '__main__':
    unittest.main()
