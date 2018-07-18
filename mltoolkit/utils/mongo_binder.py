from gridfs import GridFS, GridFSBucket
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from .file_utils import ActiveFiles
from .concepts import AutoInitAndCloseable

__all__ = ['MongoBinder']


class MongoBinder(AutoInitAndCloseable):
    """
    Base class for MongoDB data binder.

    A MongoDB data binder may save and load data in a MongoDB.
    This class provides the basic interface for accessing the MongoDB.
    """

    def __init__(self, conn_str, db_name, coll_name):
        """
        Initialize the internal states of :class:`MongoBinder`.

        Args:
            conn_str (str): The MongoDB connection string.
            db_name (str): The MongoDB database name.
            coll_name (str): The collection name (prefix) of the GridFS.
        """
        self._conn_str = conn_str
        self._db_name = db_name
        self._coll_name = coll_name
        self._fs_coll_name = '{}.files'.format(coll_name)

        self._client = None  # type: MongoClient
        self._db = None  # type: Database
        self._gridfs = None  # type: GridFS
        self._gridfs_bucket = None  # type: GridFSBucket
        self._collection = None  # type: Collection
        self._active_files = ActiveFiles()

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
