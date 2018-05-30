import unittest

import pytest
from mock import Mock

from mltoolkit.datafs import *


class ExceptionsTestCase(unittest.TestCase):

    def test_DataFileNotExist(self):
        e = DataFileNotExist('data file')
        self.assertEquals('data file', e.filename)
        self.assertEquals('Data file not exist: \'data file\'', str(e))

    def test_MetaKeyNotExist(self):
        e = MetaKeyNotExist('data file', 'meta key')
        self.assertEquals('data file', e.filename)
        self.assertEquals('meta key', e.meta_key)
        self.assertEquals('In file \'data file\': '
                          'meta key not exist: \'meta key\'', str(e))


class DataFSCapacityTestCase(unittest.TestCase):

    FLAGS = ('read_data', 'write_data', 'read_meta', 'write_meta',
             'list_meta', 'quick_count', 'random_sample')

    def test_empty(self):
        c = DataFSCapacity()
        for flag in self.FLAGS:
            self.assertFalse(getattr(c, 'can_{}'.format(flag))())
        self.assertEquals('DataFSCapacity()', repr(c))

    def test_single(self):
        for flag in self.FLAGS:
            c = DataFSCapacity(getattr(DataFSCapacity, flag.upper()))
            self.assertTrue(getattr(c, 'can_{}'.format(flag))())
            for f in self.FLAGS:
                if f != flag:
                    self.assertFalse(getattr(c, 'can_{}'.format(f))())
            self.assertEquals('DataFSCapacity({})'.format(flag), repr(c))

    def test_all(self):
        c = DataFSCapacity(DataFSCapacity.ALL)
        for flag in self.FLAGS:
            self.assertTrue(getattr(c, 'can_{}'.format(flag))())
        self.assertEquals('DataFSCapacity({})'.format(','.join(self.FLAGS)),
                          repr(c))

    def test_read_write_data(self):
        c = DataFSCapacity(DataFSCapacity.READ_WRITE_DATA)
        self.assertEquals('DataFSCapacity(read_data,write_data)', repr(c))

    def test_read_write_meta(self):
        c = DataFSCapacity(DataFSCapacity.READ_WRITE_META)
        self.assertEquals('DataFSCapacity(read_meta,write_meta)', repr(c))


class DataFSTestCase(unittest.TestCase):

    def test_props(self):
        self.assertFalse(DataFS().strict)
        self.assertTrue(DataFS(strict=True).strict)

    def test_as_flow(self):
        class DummyFS(DataFS):
            def __init__(self):
                super(DummyFS, self).__init__()
                self.clone = Mock(return_value=cloned)
                self._capacity = DataFSCapacity(DataFSCapacity.ALL)

            @property
            def capacity(self):
                return self._capacity

        cloned = object()
        fs = DummyFS()

        # as_flow with default args
        flow = fs.as_flow(123)
        self.assertIsInstance(flow, DataFSFlow)
        self.assertIs(cloned, flow.fs)
        self.assertEquals(123, flow.batch_size)
        self.assertTrue(flow.with_names)
        self.assertIsNone(flow.meta_keys)
        self.assertFalse(flow.is_shuffled)
        self.assertFalse(flow.skip_incomplete)
        self.assertIsNone(flow.names_pattern)

        # as_flow with customized args
        flow = fs.as_flow(123, with_names=False, meta_keys=['a', 'b'],
                          shuffle=True, skip_incomplete=True,
                          names_pattern='.*\.jpg$')
        self.assertIsInstance(flow, DataFSFlow)
        self.assertIs(cloned, flow.fs)
        self.assertEquals(123, flow.batch_size)
        self.assertFalse(flow.with_names)
        self.assertEquals(('a', 'b'), flow.meta_keys)
        self.assertTrue(flow.is_shuffled)
        self.assertTrue(flow.skip_incomplete)
        self.assertTrue(hasattr(flow.names_pattern, 'match'))
        self.assertEquals('.*\.jpg$', flow.names_pattern.pattern)

        # as_random_flow with default args
        flow = fs.as_random_flow(123)
        self.assertIsInstance(flow, DataFSRandomFlow)
        self.assertIs(cloned, flow.fs)
        self.assertEquals(123, flow.batch_size)
        self.assertTrue(flow.with_names)
        self.assertIsNone(flow.meta_keys)
        self.assertFalse(flow.skip_incomplete)

        # as_random_flow with customized args
        flow = fs.as_random_flow(123, with_names=False, meta_keys=['a', 'b'],
                                 skip_incomplete=True)
        self.assertIsInstance(flow, DataFSRandomFlow)
        self.assertIs(cloned, flow.fs)
        self.assertEquals(123, flow.batch_size)
        self.assertFalse(flow.with_names)
        self.assertEquals(('a', 'b'), flow.meta_keys)
        self.assertTrue(flow.skip_incomplete)

        # as_random_flow with capacity check
        fs._capacity = DataFSCapacity(
            DataFSCapacity.ALL & ~DataFSCapacity.RANDOM_SAMPLE)
        with pytest.raises(UnsupportedOperation):
            _ = fs.as_random_flow(123)


if __name__ == '__main__':
    unittest.main()
