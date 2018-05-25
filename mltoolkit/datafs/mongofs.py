from gridfs import GridFS
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from .base import DataFS, DataFSCapacity

__all__ = ['MongoFS']


class MongoFS(DataFS):

    def __init__(self, conn_str, db_name, coll_name):
        super(DataFS, self).__init__()
        self._capacity = DataFSCapacity(DataFSCapacity.ALL)

        self._conn_str = conn_str
        self._db_name = db_name
        self._coll_name = coll_name
        self._fs_coll_name = '{}.files'.format(coll_name)

        self._client = None  # type: MongoClient
        self._db = None  # type: Database
        self._gridfs = None  # type: GridFS
        self._collection = None  # type: Collection

    @property
    def conn_str(self):
        return self._conn_str

    @property
    def db_name(self):
        return self._db_name

    @property
    def coll_name(self):
        return self._coll_name

    @property
    def client(self):
        self.init()
        return self._client

    @property
    def db(self):
        self.init()
        return self._db

    @property
    def gridfs(self):
        self.init()
        return self._gridfs

    @property
    def collection(self):
        return self._collection

    def _init(self):
        self._client = MongoClient(self._conn_str)
        self._db = self._client.get_database(self._db_name)
        self._gridfs = GridFS(self._db, self._coll_name)
        self._collection = self._db[self._coll_name]

    def _close(self):
        try:
            if self._client is not None:
                self._client.close()
        finally:
            self._gridfs = None
            self._db = None
            self._client = None

    def clone(self):
        return MongoFS(self.conn_str, self.db_name, self.coll_name)

    @property
    def capacity(self):
        return self._capacity

    def count(self):
        return self.collection.files.count()

    def iter_names(self):
        for r in self.collection.files.find({}, {'filename': 1, '_id': 0}):
            yield r['filename']

    def sample_names(self, n_samples):
        pass

    def iter_files(self, meta_keys=None):
        doc_filter = {'filename': 1, '_id': 1}
        if meta_keys is not None:
            meta_keys = tuple(meta_keys)
            doc_filter.update({k: 1 for k in meta_keys or ()})
            for r in self.gridfs.find({}, doc_filter, no_cursor_timeout=True):
                yield (r['filename'], r.read()) + tuple(r[k] for k in meta_keys)
        else:
            for r in self.gridfs.find({}, doc_filter, no_cursor_timeout=True):
                yield (r['filename'], r.read())

    def sample_files(self, n_samples, meta_keys=None):
        pass

    def retrieve(self, filename, meta_keys=None):
        pass

    def get_data(self, filename):
        pass

    def put_data(self, filename, data):
        pass

    def open(self, filename, mode):
        pass

    def list_meta(self, filename):
        pass

    def get_meta(self, filename, meta_keys):
        pass

    def batch_get_meta(self, filenames, meta_keys):
        pass

    def get_meta_dict(self, filename):
        pass

    def put_meta(self, filename, meta_dict=None, **meta_dict_kwargs):
        pass

    def clear_and_put_meta(self, filename, meta_dict=None, **meta_dict_kwargs):
        pass

    def clear_meta(self, filename):
        pass
