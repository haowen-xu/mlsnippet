import gc
import subprocess
import unittest
import uuid
from contextlib import contextmanager
from io import BytesIO

import six
from gridfs import GridFS, GridFSBucket
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from mltoolkit.datafs import *
from .standard_checks import StandardFSChecks


class MongoFSTestCase(unittest.TestCase, StandardFSChecks):

    def get_snapshot(self, fs):
        conn_str = 'mongodb://root:123456@127.0.0.1:27017/admin'
        client = MongoClient(conn_str)
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
        daemon_name = uuid.uuid4().hex
        subprocess.check_call([
            'docker', 'run', '--rm', '-d',
            '--name', daemon_name,
            '-e', 'MONGO_INITDB_ROOT_USERNAME=root',
            '-e', 'MONGO_INITDB_ROOT_PASSWORD=123456',
            '-p', '27017:27017',
            'mongo'
        ])
        print('Docker daemon started: {!r}'.format(daemon_name))
        try:
            conn_str = 'mongodb://root:123456@127.0.0.1:27017/admin'
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
        finally:
            subprocess.check_call(['docker', 'kill', daemon_name])

    def test_standard(self):
        self.run_standard_checks(DataFSCapacity.ALL)

    def test_mongofs_props_and_methods(self):
        with self.temporary_fs() as fs:
            # test auto-init
            self.assertIsInstance(fs.clone().client, MongoClient)
            self.assertIsInstance(fs.clone().db, Database)
            self.assertIsInstance(fs.clone().gridfs, GridFS)
            self.assertIsInstance(fs.clone().gridfs_bucket, GridFSBucket)
            self.assertIsInstance(fs.clone().collection, Collection)
            gc.collect()


if __name__ == '__main__':
    unittest.main()
