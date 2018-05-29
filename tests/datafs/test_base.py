import unittest

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


if __name__ == '__main__':
    unittest.main()
