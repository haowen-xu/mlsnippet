from gridfs import GridFS, GridFSBucket
from pymongo import MongoClient, CursorType
from pymongo.database import Database
from pymongo.collection import Collection

from .base import DataFS, DataFSCapacity
from .errors import DataFileNotExist, InvalidOpenMode, MetaKeyNotExist
from .utils import ActiveFiles

__all__ = ['MongoFS']

META_FIELD = 'metadata'


class MongoFS(DataFS):
    """
    MongoDB GridFS based :class:`DataFS`.

    This class provides a :class:`DataFS`, which saves the files in
    a MongoDB GridFS, and stores the meta values in ``metadata`` field
    of each record in the fs collection.
    """

    def __init__(self, conn_str, db_name, coll_name, strict=False):
        """
        Construct a new :class:`MongoFS`.

        Args:
            conn_str (str): The MongoDB connection string.
            db_name (str): The MongoDB database name.
            coll_name (str): The collection name (prefix) of the GridFS.
            strict (bool): Whether or not this :class:`DataFS` works in
                strict mode?  (default :obj:`False`)
        """
        super(MongoFS, self).__init__(
            capacity=DataFSCapacity.ALL,
            strict=strict
        )

        self._conn_str = conn_str
        self._db_name = db_name
        self._coll_name = coll_name
        self._fs_coll_name = '{}.files'.format(coll_name)

        self._client = None  # type: MongoClient
        self._db = None  # type: Database
        self._gridfs = None  # type: GridFS
        self._gridfs_bucket = None  # type: GridFSBucket
        self._collection = None  # type: Collection

        if self.strict:
            def get_meta_value(r, m, k):
                if k not in m:
                    raise MetaKeyNotExist(r['filename'], k)
                return m[k]
        else:
            get_meta_value = lambda r, m, k: m.get(k)

        self._get_meta_value_from_record = get_meta_value
        self._active_files = ActiveFiles()

    def _make_query_project(self, meta_keys=None, _id=1, filename=1):
        ret = {'_id': _id, 'filename': filename}
        if meta_keys:
            ret.update({'{}.{}'.format(META_FIELD, k): 1
                        for k in meta_keys})
        return ret

    def _make_sample_cursor(self, n_samples, meta_keys, with_id=1):
        project = self._make_query_project(_id=with_id, meta_keys=meta_keys)
        return self.collection.files.aggregate([
            {'$sample': {'size': n_samples}},
            {'$project': project},
        ])

    def _make_result_meta(self, record, meta_keys):
        meta_dict = record.get(META_FIELD)
        if not meta_dict or not isinstance(meta_dict, dict):
            meta_dict = {}
        return tuple(self._get_meta_value_from_record(record, meta_dict, k)
                     for k in meta_keys)

    @property
    def conn_str(self):
        """Get the MongoDB connection string."""
        return self._conn_str

    @property
    def db_name(self):
        """Get the MongoDB database name."""
        return self._db_name

    @property
    def coll_name(self):
        """Get the collection name (prefix) of the GridFS."""
        return self._coll_name

    @property
    def client(self):
        """
        Get the MongoDB client.  Reading this property will force
        the internal states of :class:`MongoFS` to be initialized.

        Returns:
            MongoClient: The MongoDB client.
        """
        self.init()
        return self._client

    @property
    def db(self):
        """
        Get the MongoDB database object.  Reading this property will force
        the internal states of :class:`MongoFS` to be initialized.

        Returns:
            Database: The MongoDB database object.
        """
        self.init()
        return self._db

    @property
    def gridfs(self):
        """
        Get the MongoDB GridFS client.  Reading this property will force
        the internal states of :class:`MongoFS` to be initialized.

        Returns:
            GridFS: The MongoDB GridFS client.
        """
        self.init()
        return self._gridfs

    @property
    def gridfs_bucket(self):
        """
        Get the MongoDB GridFS bucket.  Reading this property will force
        the internal states of :class:`MongoFS` to be initialized.

        Returns:
            GridFSBucket: The MongoDB GridFS bucket.
        """
        self.init()
        return self._gridfs_bucket

    @property
    def collection(self):
        """
        Get the MongoDB collection object.  Reading this property will force
        the internal states of :class:`MongoFS` to be initialized.

        Returns:
            Collection: The MongoDB collection object.
        """
        self.init()
        return self._collection

    def _init(self):
        self._client = MongoClient(self._conn_str)
        self._db = self._client.get_database(self._db_name)
        self._collection = self._db[self._coll_name]
        self._gridfs = GridFS(self._db, self._coll_name)
        self._gridfs_bucket = GridFSBucket(self._db, self._coll_name)

    def _close(self):
        self._active_files.close_all()
        try:
            if self._client is not None:
                self._client.close()
        finally:
            self._gridfs = None
            self._db = None
            self._client = None

    def clone(self):
        return MongoFS(self.conn_str, self.db_name, self.coll_name,
                       strict=self.strict)

    def count(self):
        return self.collection.files.count()

    def iter_names(self):
        for r in self.collection.files.find({}, {'filename': 1, '_id': 0}):
            yield r['filename']

    def sample_names(self, n_samples):
        return [r['filename']
                for r in self._make_sample_cursor(n_samples, (), 0)]

    def iter_files(self, meta_keys=None):
        meta_keys = tuple(meta_keys or ())
        project = self._make_query_project(meta_keys)
        for r in self.collection.files.find({}, project,
                                            no_cursor_timeout=True,
                                            cursor_type=CursorType.EXHAUST):
            with self.gridfs.get(r['_id']) as f:
                data = f.read()
            yield (r['filename'], data) + self._make_result_meta(r, meta_keys)

    def sample_files(self, n_samples, meta_keys=None):
        meta_keys = tuple(meta_keys or ())
        ret = []
        for r in self._make_sample_cursor(n_samples, meta_keys, 1):
            with self.gridfs.get(r['_id']) as f:
                data = f.read()
            ret.append(
                (r['filename'], data) + self._make_result_meta(r, meta_keys))
        return ret

    def retrieve(self, filename, meta_keys=None):
        has_meta_keys = meta_keys is not None
        meta_keys = tuple(meta_keys or ())
        project = self._make_query_project(meta_keys)
        r = self.collection.files.find_one(
            {'filename': filename}, project, no_cursor_timeout=True)
        if r is None:
            raise DataFileNotExist(filename)
        with self.gridfs.get(r['_id']) as f:
            data = f.read()
        if has_meta_keys:
            return (data,) + self._make_result_meta(r, meta_keys)
        else:
            return data

    def put_data(self, filename, data):
        f = self.collection.files.find_one({'filename': filename}, {'_id': 1})
        if f is not None:
            self.gridfs.delete(f['_id'])
        self.gridfs.put(data, filename=filename)

    def open(self, filename, mode):
        f = self.collection.files.find_one({'filename': filename}, {'_id': 1})
        if mode == 'r':
            if f is None:
                raise DataFileNotExist(filename)
            return self._active_files.add(
                self.gridfs_bucket.open_download_stream(f['_id']))
        elif mode == 'w':
            if f is not None:
                self.gridfs.delete(f['_id'])
            return self._active_files.add(
                self.gridfs_bucket.open_upload_stream(filename))
        else:
            raise InvalidOpenMode(mode)

    def isfile(self, filename):
        return self.gridfs.exists({'filename': filename})

    def batch_isfile(self, filenames):
        filenames = tuple(filenames)
        ret = [False] * len(filenames)
        name_to_id = {filename: i for i, filename in enumerate(filenames)}
        for f in self.collection.files.find({'filename': {"$in": filenames}},
                                            {'filename': 1, '_id': 0}):
            ret[name_to_id[f['filename']]] = True
        return ret

    def list_meta(self, filename):
        return tuple(self.get_meta_dict(filename).keys())

    def get_meta(self, filename, meta_keys):
        meta_keys = tuple(meta_keys or ())
        project = self._make_query_project(meta_keys, _id=0, filename=0)
        r = self.collection.files.find_one({'filename': filename}, project)
        if r is None:
            raise DataFileNotExist(filename)
        return self._make_result_meta(r, meta_keys)

    def batch_get_meta(self, filenames, meta_keys):
        filenames = tuple(filenames)
        meta_keys = tuple(meta_keys or ())
        project = self._make_query_project(meta_keys, _id=0, filename=1)
        ret = [None] * len(filenames)
        name_to_id = {filename: i for i, filename in enumerate(filenames)}
        for r in self.collection.files.find({'filename': {'$in': filenames}},
                                            project):
            ret[name_to_id[r['filename']]] = \
                self._make_result_meta(r, meta_keys)
        return ret

    def get_meta_dict(self, filename):
        f = self.collection.files.find_one(
            {'filename': filename}, {META_FIELD: 1})
        if f is None:
            raise DataFileNotExist(filename)
        meta_dict = f.get(META_FIELD, None)
        if not meta_dict or not isinstance(meta_dict, dict):
            return {}
        else:
            return meta_dict

    def _merge_meta_dict(self, d1, d2):
        if not d1 and not d2:
            ret = {}
        elif not d1:
            ret = dict(d2)
        elif not d2:
            ret = dict(d1)
        else:
            ret = dict(d1)
            ret.update(dict(d2))
        return ret

    def put_meta(self, filename, meta_dict=None, **meta_dict_kwargs):
        meta_dict = self._merge_meta_dict(meta_dict, meta_dict_kwargs)
        if meta_dict:
            merged_dict = self.get_meta_dict(filename)
            merged_dict.update(meta_dict)
            self.clear_and_put_meta(filename, merged_dict)

    def clear_and_put_meta(self, filename, meta_dict=None, **meta_dict_kwargs):
        meta_dict = self._merge_meta_dict(meta_dict, meta_dict_kwargs)
        update = ({'$set': {'metadata': meta_dict}}
                  if meta_dict else {'$unset': {'metadata': 1}})
        r = self.collection.files.update_one({'filename': filename}, update)
        if not r.modified_count:
            if not self.isfile(filename):
                raise DataFileNotExist(filename)

    def clear_meta(self, filename):
        self.clear_and_put_meta(filename)
