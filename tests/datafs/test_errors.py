import unittest

from mlsnippet.datafs import DataFileNotExist, MetaKeyNotExist, InvalidOpenMode


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

    def test_InvalidOpenMode(self):
        e = InvalidOpenMode('invalid mode')
        self.assertEquals('invalid mode', e.mode)
        self.assertEquals('Invalid open mode: \'invalid mode\'', str(e))


if __name__ == '__main__':
    unittest.main()
