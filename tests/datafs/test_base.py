import os
import random
import unittest
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import six
import pytest
from mock import Mock

from mltoolkit.datafs import *
from mltoolkit.datafs import UnsupportedOperation, DataFileNotExist
from mltoolkit.utils import TemporaryDirectory, makedirs, iter_files
from .standard_checks import StandardFSChecks, LocalFS
from .test_dataflow import _DummyDataFS


class DataFSCapacityTestCase(unittest.TestCase):

    FLAGS = ('read_data', 'write_data', 'read_meta', 'write_meta',
             'list_meta', 'quick_count', 'random_sample')

    def test_empty(self):
        c = DataFSCapacity()
        c2 = DataFSCapacity()
        for flag in self.FLAGS:
            self.assertFalse(getattr(c, 'can_{}'.format(flag))())
        self.assertEquals('DataFSCapacity()', repr(c))
        self.assertEquals(c, c2)
        self.assertEquals(hash(c), hash(c2))
        for flag in self.FLAGS:
            c3 = DataFSCapacity(getattr(DataFSCapacity, flag.upper()))
            self.assertNotEquals(c, c3)

    def test_single(self):
        for flag in self.FLAGS:
            c = DataFSCapacity(getattr(DataFSCapacity, flag.upper()))
            c2 = DataFSCapacity(getattr(DataFSCapacity, flag.upper()))
            self.assertTrue(getattr(c, 'can_{}'.format(flag))())
            self.assertEquals(c, c2)
            self.assertEquals(hash(c), hash(c2))
            for f in self.FLAGS:
                if f != flag:
                    c3 = DataFSCapacity(getattr(DataFSCapacity, f.upper()))
                    self.assertFalse(getattr(c, 'can_{}'.format(f))())
                    self.assertNotEquals(c, c3)
            self.assertEquals('DataFSCapacity({})'.format(flag), repr(c))

    def test_all(self):
        c = DataFSCapacity(DataFSCapacity.ALL)
        c2 = DataFSCapacity(DataFSCapacity.ALL)
        for flag in self.FLAGS:
            self.assertTrue(getattr(c, 'can_{}'.format(flag))())
        self.assertEquals('DataFSCapacity({})'.format(','.join(self.FLAGS)),
                          repr(c))
        self.assertEquals(c, c2)
        self.assertEquals(hash(c), hash(c2))

    def test_read_write_data(self):
        c = DataFSCapacity(DataFSCapacity.READ_WRITE_DATA)
        self.assertEquals('DataFSCapacity(read_data,write_data)', repr(c))

    def test_read_write_meta(self):
        c = DataFSCapacity(DataFSCapacity.READ_WRITE_META)
        self.assertEquals('DataFSCapacity(read_meta,write_meta)', repr(c))

    def test_construction_from_DataFSCapacity(self):
        c = DataFSCapacity(DataFSCapacity.ALL)
        c2 = DataFSCapacity(c)
        self.assertIsInstance(c2, DataFSCapacity)
        self.assertEquals(c, c2)


class DataFSTestCase(unittest.TestCase):

    def test_props(self):
        self.assertFalse(DataFS(DataFSCapacity()).strict)
        self.assertTrue(DataFS(DataFSCapacity(), strict=True).strict)

    def test_as_flow(self):
        fs = _DummyDataFS()
        fs.clone = Mock(wraps=fs.clone)

        # as_flow with default args, returns DataFSForwardFlow
        flow = fs.as_flow(123)
        self.assertIsInstance(flow, DataFSForwardFlow)
        self.assertIsNot(fs, flow.fs)
        self.assertEquals(1, fs.clone.call_count)
        self.assertEquals(123, flow.batch_size)
        self.assertTrue(flow.with_names)
        self.assertIsNone(flow.meta_keys)
        self.assertFalse(flow.skip_incomplete)

        # as_flow with custom args, still DataFSForwardFlow
        flow = fs.as_flow(123, with_names=False, meta_keys=iter('abcd'),
                          skip_incomplete=True)
        self.assertIsInstance(flow, DataFSForwardFlow)
        self.assertIsNot(fs, flow.fs)
        self.assertEquals(2, fs.clone.call_count)
        self.assertEquals(123, flow.batch_size)
        self.assertFalse(flow.with_names)
        self.assertEquals(('a', 'b', 'c', 'd'), flow.meta_keys)
        self.assertTrue(flow.skip_incomplete)

        # as_flow with custom args, DataFSIndexedFlow
        flow = fs.as_flow(123, with_names=False, meta_keys=iter('abcd'),
                          shuffle=True, skip_incomplete=True)
        self.assertIsInstance(flow, DataFSIndexedFlow)
        self.assertIsNot(fs, flow.fs)
        self.assertEquals(3, fs.clone.call_count)
        self.assertEquals(123, flow.batch_size)
        self.assertFalse(flow.with_names)
        self.assertTrue(flow.is_shuffled)
        np.testing.assert_equal(fs.list_names(), flow.names)
        self.assertEquals(('a', 'b', 'c', 'd'), flow.meta_keys)
        self.assertTrue(flow.skip_incomplete)

        # as_flow with custom args, DataFSIndexedFlow
        flow = fs.as_flow(123, with_names=False, meta_keys=iter('abcd'),
                          names_pattern='^[034589]$', skip_incomplete=True)
        self.assertIsInstance(flow, DataFSIndexedFlow)
        self.assertIsNot(fs, flow.fs)
        self.assertEquals(4, fs.clone.call_count)
        self.assertEquals(123, flow.batch_size)
        self.assertFalse(flow.with_names)
        self.assertFalse(flow.is_shuffled)
        np.testing.assert_equal(list('034589'), flow.names)
        self.assertEquals(('a', 'b', 'c', 'd'), flow.meta_keys)
        self.assertTrue(flow.skip_incomplete)

    def test_sub_flow(self):
        fs = _DummyDataFS()
        fs.clone = Mock(wraps=fs.clone)
        names = list('034589')

        # sub_flow with default args
        flow = fs.sub_flow(123, names)
        self.assertIsInstance(flow, DataFSIndexedFlow)
        self.assertIsNot(fs, flow.fs)
        self.assertEquals(1, fs.clone.call_count)
        self.assertEquals(123, flow.batch_size)
        self.assertTrue(flow.with_names)
        self.assertFalse(flow.is_shuffled)
        np.testing.assert_equal(names, flow.names)
        self.assertFalse(flow.skip_incomplete)

        # sub_flow with custom args
        flow = fs.sub_flow(123, names, with_names=False, meta_keys=iter('abcd'),
                           shuffle=True, skip_incomplete=True)
        self.assertIsInstance(flow, DataFSIndexedFlow)
        self.assertIsNot(fs, flow.fs)
        self.assertEquals(2, fs.clone.call_count)
        self.assertEquals(123, flow.batch_size)
        self.assertFalse(flow.with_names)
        self.assertTrue(flow.is_shuffled)
        np.testing.assert_equal(list('034589'), flow.names)
        self.assertEquals(('a', 'b', 'c', 'd'), flow.meta_keys)
        self.assertTrue(flow.skip_incomplete)

    def test_random_flow(self):
        fs = _DummyDataFS()
        fs.clone = Mock(wraps=fs.clone)
        fs._capacity = DataFSCapacity(DataFSCapacity.ALL)

        # as_random_flow with default args
        flow = fs.random_flow(123)
        self.assertIsInstance(flow, DataFSRandomFlow)
        self.assertIsNot(fs, flow.fs)
        self.assertEquals(123, flow.batch_size)
        self.assertTrue(flow.with_names)
        self.assertIsNone(flow.meta_keys)
        self.assertFalse(flow.skip_incomplete)

        # as_random_flow with customized args
        flow = fs.random_flow(123, with_names=False, meta_keys=['a', 'b'],
                              skip_incomplete=True)
        self.assertIsInstance(flow, DataFSRandomFlow)
        self.assertIsNot(fs, flow.fs)
        self.assertEquals(123, flow.batch_size)
        self.assertFalse(flow.with_names)
        self.assertEquals(('a', 'b'), flow.meta_keys)
        self.assertTrue(flow.skip_incomplete)

        # as_random_flow with capacity check
        fs._capacity = DataFSCapacity(
            DataFSCapacity.ALL & ~DataFSCapacity.RANDOM_SAMPLE)
        with pytest.raises(UnsupportedOperation):
            _ = fs.random_flow(123)


class ExtendedLocalFS(LocalFS):
    """
    Extended :class:`LocalFS`, for testing inherited methods from
    :class:`DataFS`.
    """

    def __init__(self, *args, **kwargs):
        super(ExtendedLocalFS, self).__init__(*args, **kwargs)
        self._file_meta_dict = defaultdict(lambda: {})
        self._capacity = DataFSCapacity(
            DataFSCapacity.ALL & ~DataFSCapacity.QUICK_COUNT)

    def clone(self):
        return ExtendedLocalFS(self.root_dir, strict=self.strict)

    def sample_names(self, n_samples):
        return [random.choice(self.list_names()) for _ in range(n_samples)]

    def list_meta(self, filename):
        if not self.isfile(filename):
            raise DataFileNotExist(filename)
        return tuple(self._file_meta_dict[filename].keys())

    def get_meta(self, filename, meta_keys):
        meta_keys = tuple(meta_keys or ())
        if not self.isfile(filename):
            raise DataFileNotExist(filename)
        meta_dict = self._file_meta_dict[filename]
        if self.strict:
            for k in meta_keys:
                if k not in meta_dict:
                    raise MetaKeyNotExist(filename, k)
        return tuple(meta_dict.get(k, None) for k in meta_keys)

    def put_meta(self, filename, meta_dict=None, **meta_dict_kwargs):
        if not self.isfile(filename):
            raise DataFileNotExist(filename)
        merged = {}
        if meta_dict:
            merged.update(dict(meta_dict))
        if meta_dict_kwargs:
            merged.update(meta_dict_kwargs)
        self._file_meta_dict[filename].update(merged)

    def clear_meta(self, filename):
        if not self.isfile(filename):
            raise DataFileNotExist(filename)
        self._file_meta_dict[filename] = {}


class ExtendedLocalFSTestCase(unittest.TestCase, StandardFSChecks):

    def get_snapshot(self, fs):
        ret = {}
        for name in iter_files(fs.root_dir):
            with open(os.path.join(fs.root_dir, name), 'rb') as f:
                cnt = f.read()
            meta_dict = fs._file_meta_dict[name]
            if meta_dict:
                ret[name] = (cnt, meta_dict)
            else:
                ret[name] = (cnt,)
        return ret

    @contextmanager
    def temporary_fs(self, snapshot=None, **kwargs):
        with TemporaryDirectory() as tempdir:
            fs = ExtendedLocalFS(tempdir, **kwargs)
            if snapshot:
                for filename, payload in six.iteritems(snapshot):
                    content = payload[0]
                    meta_dict = payload[1] if len(payload) > 1 else None
                    file_path = os.path.join(tempdir, filename)
                    file_dir = os.path.split(file_path)[0]
                    makedirs(file_dir, exist_ok=True)
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    if meta_dict:
                        fs.put_meta(filename, meta_dict)
            with fs:
                yield fs

    def test_standard(self):
        self.run_standard_checks(
            DataFSCapacity.ALL & ~DataFSCapacity.QUICK_COUNT)


if __name__ == '__main__':
    unittest.main()
