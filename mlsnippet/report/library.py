import weakref

from mlsnippet.utils import DocInherit

__all__ = ['Library', 'SingletonLibrary', 'JQuery', 'MathJax']


@DocInherit
class Library(object):
    """
    Base class for HTML libraries.

    An HTML library carries stylesheets and scripts, which will be output
    at the end of HEAD (stylesheets) and BODY (scripts).
    """

    def get_styles(self):
        """
        Get the stylesheets of this library.

        Derived classes should override this method to provide the source
        code of the stylesheets, including the "<style>" or "<link>" tag.
        For example:

        .. code-block:: python

            class Bootstrap(Library):

                def get_styles(self):
                    return '<link rel="stylesheet" ' \
                        'href="https://cdnjs.cloudflare.com/ajax/libs/' \
                        'twitter-bootstrap/3.3.7/css/bootstrap.min.css" />'

        The stylesheets will be output at the end of HEAD tag in the rendered
        HTML.

        Returns:
            str or None: Source code of the stylesheets, or :obj:`None`
                if no stylesheet is provided.
        """
        return None

    def get_scripts(self):
        """
        Get the scripts of this library.

        Derived classes should override this method to provide the source
        code of the scripts, including the "<script>" tag.  For example:

        .. code-block:: python

            class JQuery(Library):

                def get_scripts(self):
                    return '<script src="https://cdnjs.cloudflare.com/ajax/' \
                        'libs/jquery/3.2.1/jquery.min.js"></script>'

        The scripts will be output at the end of BODY tag in the rendered HTML.

        Returns:
            str or None: Source code of the scripts, or :obj:`None`
                if no script is provided.
        """
        return None


class SingletonLibrary(Library):
    """
    Base for singleton :class:`Library` classes.

    Subclasses of :class:`SingletonLibrary` would become singleton.
    For example, :class:`JQuery` is a subclass of :class:`SingletonLibrary`,
    such that the following assertion holds:

    .. code-block:: python

        assert(JQuery() is JQuery())
    """

    def __new__(cls, *args, **kwargs):
        if cls not in _singleton_library_instances:
            _singleton_library_instances[cls] = \
                super(Library, cls).__new__(cls, *args, **kwargs)
        return _singleton_library_instances[cls]


# Dict of :class:`SingletonLibrary` subclasses instances
_singleton_library_instances = weakref.WeakKeyDictionary()


class JQuery(SingletonLibrary):
    """JQuery library, loaded via ``requirejs(["jquery"])``."""

    def get_scripts(self):
        return '<script type="text/javascript">requirejs(["jquery"]);</script>'


class MathJax(SingletonLibrary):
    """MathJax library, loaded via ``requirejs(["mathjax"])``."""

    def get_scripts(self):
        return '<script type="text/javascript">requirejs(["mathjax"]);</script>'
