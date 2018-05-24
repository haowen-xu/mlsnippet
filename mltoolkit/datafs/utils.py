import weakref

__all__ = ['ActiveFiles']


class ActiveFiles(object):

    def __init__(self):
        self._files = weakref.WeakSet()

    def add(self, file_obj):
        self._files.add(file_obj)
        return file_obj

    def close_all(self):
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass
        self._files.clear()
