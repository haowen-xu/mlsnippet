import gc
import unittest
from contextlib import contextmanager
from io import BytesIO

import six
from gridfs import GridFS, GridFSBucket
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from mlsnippet.datafs import *
from mlsnippet.utils import maybe_close
from .standard_checks import StandardFSChecks
from ..helper import temporary_mongodb


class MongoFSTestCase(unittest.TestCase, StandardFSChecks):

    def get_snapshot(self, fs):
        client = MongoClient(fs.conn_str)
        try:
            ret = {}
            database = client.get_database('admin')
            files_coll = database['test.files']
            gridfs = GridFS(database, 'test')

            for r in files_coll.find({}):
                name = r['filename']
                with maybe_close(gridfs.get(r['_id'])) as f:
                    cnt = f.read()
                if 'metadata' in r:
                    ret[name] = (cnt, r['metadata'])
                else:
                    ret[name] = (cnt,)
            return ret
        finally:
            client.close()

    @contextmanager
    def temporary_fs(self, snapshot=None, **kwargs):
        with temporary_mongodb() as conn_str:
            if snapshot:
                client = MongoClient(conn_str)
                try:
                    database = client.get_database('admin')
                    files_coll = database['test.files']
                    gridfs_bucket = GridFSBucket(database, 'test')
                    for filename, payload in six.iteritems(snapshot):
                        content = payload[0]
                        meta_dict = payload[1] if len(payload) > 1 else None
                        gridfs_bucket.upload_from_stream(
                            filename,
                            BytesIO(content)
                        )
                        if meta_dict:
                            files_coll.update_one(
                                {'filename': filename},
                                {'$set': {'metadata': meta_dict}}
                            )
                finally:
                    client.close()
            with MongoFS(conn_str, 'admin', 'test', **kwargs) as fs:
                yield fs

    def test_standard(self):
        self.run_standard_checks(DataFSCapacity.ALL)

    def test_mongofs_props_and_methods(self):
        with self.temporary_fs() as fs:
            # test auto-init on cloned objects
            self.assertIsInstance(fs.clone().client, MongoClient)
            self.assertIsInstance(fs.clone().db, Database)
            self.assertIsInstance(fs.clone().gridfs, GridFS)
            self.assertIsInstance(fs.clone().gridfs_bucket, GridFSBucket)
            self.assertIsInstance(fs.clone().collection, Collection)
            gc.collect()  # cleanup cloned objects


if __name__ == '__main__':
    unittest.main()
