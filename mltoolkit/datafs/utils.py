import sys
import weakref

import six

__all__ = ['ActiveFiles']


class ActiveFiles(object):
    """
    A set to track active (still in-use) file objects.

    This class holds weak references to active file objects, which will be
    all closed when :meth:`close_all` of this class is called.

    Since the references are weak, once a file object is no longer referenced
    by any other caller, it will be automatically removed from this set.
    However, in this situation, the :meth:`close` method of the de-referenced
    file object will not be called by this class.

    Such a class is majorly designed for keeping track of active file objects
    opened by a :class:`~mltoolkit.datafs.DataFS`, which are forced to be
    closed as soon as the :class:`~mltoolkit.datafs.DataFS` is closed.
    """

    def __init__(self):
        self._files = weakref.WeakSet()

    def add(self, file_obj):
        """
        Add a file object to this set.

        Args:
            file_obj: The file object to be tracked.

        Returns:
            The provided `file_obj`.
        """
        self._files.add(file_obj)
        return file_obj

    def close_all(self):
        """
        Close all file objects which are still active.

        All ``close()`` methods of the active file objects are ensured
        to be called.  Any error raised by the ``close()`` method of a
        file object would be catched and totally ignored, except for
        :class:`KeyboardInterrupt` and :class:`SystemExit`, which will
        be re-raised after all file objects have been closed.
        """
        reraise_buf = []
        for f in self._files:
            try:
                f.close()
            except KeyboardInterrupt:
                reraise_buf.append(sys.exc_info())
            except SystemExit:
                reraise_buf.append(sys.exc_info())
            except Exception:
                pass
        self._files.clear()
        if reraise_buf:
            six.reraise(*reraise_buf[-1])
