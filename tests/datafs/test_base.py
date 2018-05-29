import unittest

from mltoolkit.datafs import *


class DataFileNotExistTestCase(unittest.TestCase):

    def test_construction(self):
        e = DataFileNotExist('data file')
        self.assertEquals('data file', e.filename)


if __name__ == '__main__':
    unittest.main()
