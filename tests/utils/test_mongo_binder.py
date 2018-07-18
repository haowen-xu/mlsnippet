import unittest

from gridfs import GridFS, GridFSBucket
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from mltoolkit.utils import *
from ..helper import temporary_mongodb


class MongoBinderTestCase(unittest.TestCase):

    def test_props_and_methods(self):
        with temporary_mongodb() as conn_str:
            binder = MongoBinder(conn_str, 'test', 'fs')

            # test auto-init
            client = binder.client
            self.assertIsInstance(binder.client, MongoClient)
            self.assertIsInstance(binder.db, Database)
            self.assertIsInstance(binder.gridfs, GridFS)
            self.assertIsInstance(binder.gridfs_bucket, GridFSBucket)
            self.assertIsInstance(binder.collection, Collection)

            # test with context
            with binder:
                self.assertIs(client, binder.client)
            self.assertIsNone(binder._client)

            # test re-init
            self.assertIsInstance(binder.client, MongoClient)
            self.assertIsNot(client, binder.client)


if __name__ == '__main__':
    unittest.main()
