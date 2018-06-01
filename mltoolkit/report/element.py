import base64
import codecs
import mimetypes
import os
import re
from io import BytesIO

from matplotlib import pyplot
import pandas as pd
import six

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None

from mltoolkit.utils import DocInherit
from .context import ToHtmlContext
from .library import Library, MathJax
from .utils import escape


__all__ = [
    'Element', 'as_element', 'register_as_element', 'NamedElement',
    'Text', 'HTML', 'MagicHTML', 'InlineMath', 'BlockMath',
    'Resource', 'Attachment', 'Image',
    'DataFrameTable',
]


@DocInherit
class Element(object):
    """Base class for report elements."""

    def _repr_html_(self):
        """
        Render this :class:`Element` into HTML in a Jupyter notebook.

        Returns:
            str: The rendered HTML.
        """
        css_path = os.path.join(
            os.path.split(os.path.abspath(__file__))[0],
            'templates/ipython.css'
        )
        with codecs.open(css_path, 'rb', 'utf-8') as f:
            css = f.read()
        return (
            '<div class="mltoolkit-element">'
            '<style>' + css + '</style>' +
            self.to_html(ToHtmlContext(is_ipython=True)) +
            '</div>'
        )

    def to_html(self, context=None):
        """
        Render this :class:`Element` into HTML.

        Args:
            context (ToHtmlContext): Context for rendering :class:`Element`
                into HTML.  If not specified, will create a default context.

        Returns:
            str: The rendered HTML.
        """
        if context is None:
            context = ToHtmlContext()
        with context.push(self):
            return self._to_html(context)

    def _to_html(self, context):
        """
        Actual method to render this :class:`Element` into HTML.
        Subclasses of :class:`Element` should override this method.

        This method is designed to be called by :meth:`to_html`, with
        `context` ensured to be created, and `self` pushed to the
        call-stack of `context.
        Note that if an element needs to render another element within
        this method, it should call :meth:`to_html` of that element,
        with `context` passed to it.

        Args:
            context (ToHtmlContext): The context for rendering.

        Returns:
            str: The rendered HTML.
        """
        raise NotImplementedError()

    def get_libraries(self):
        """
        Get the libraries used by this element.

        Returns:
            list[Library] or None: The list of :class:`Library`, or
                :obj:`None` if no :class:`Library` is used.
        """
        return None


def as_element(obj):
    """
    Convert `obj` into :class:`Element`.

    It will be converted according the following rules in order:

    1.  Return `obj` itself if it is a :class:`Element`.
    2.  A :class:`HTML` will be constructed if `obj` is a :class:`str`.
    3.  The `conversion function` will be called, if the type of `obj`
        has been designated a function via :func:`register_as_element`.
        The registered conversion functions will be tried in the order
        of registration.  The first function which can successfully
        convert the object is selected.
    4.  A :class:`MagicHTML` will be constructed if `obj` has a magic
        method `__html__`.

    Args:
        obj: The object to be converted.

    Returns:
        Element: An instance of :class:`Element` converted from `obj`.

    Raises:
        TypeError: If `obj` cannot be converted by any of the above rules.
    """
    if isinstance(obj, Element):
        return obj
    if isinstance(obj, six.string_types):
        return HTML(obj)
    for typ_, convert_func in __as_element_convert_functions:
        if isinstance(obj, typ_):
            return convert_func(obj)
    if hasattr(obj, '__html__'):
        return MagicHTML(obj)
    raise TypeError('Cannot convert a `{}` object into `Element`: '
                    'no conversion function is registered'.
                    format(obj.__class__.__name__))


def register_as_element(typ_, convert_func):
    """
    Register a `conversion function` which converts objects of particular
    type into :class:`Element`.

    Args:
        typ_: The type which this conversion function is designed for.
        convert_func: The conversion function which converts objects of
            `type_` into :class:`Element`.

    Raises:
        TypeError: If `typ_` is not a class.
    """
    if not isinstance(typ_, six.class_types):
        raise TypeError('`typ_` must be a class')
    __as_element_convert_functions.append((typ_, convert_func))


__as_element_convert_functions = []


class NamedElement(Element):
    """Base class for :class:`Element` with name."""

    def __init__(self, name=None):
        """
        Construct the :class:`NamedElement`.

        Args:
            name (str or None): If specified, it will be used as the name
                of this element.
        """
        if name is not None:
            if not re.match(r'^[A-Za-z0-9_\-]+$', name):
                raise ValueError(r'`name` must be a non-empty string matching '
                                 r'pattern "^[A-Za-z0-9_\-]+$"')
        self._name = name

    @property
    def name(self):
        """
        Get the name of this element.

        Returns:
            str or None: The name specified in construction, or :obj:`None`
                if not specified.
        """
        return self._name


class Text(Element):
    """Plain text element."""

    def __init__(self, text):
        """
        Construct the :class:`Text`.

        Args:
            text (str): The text content.  The HTML entities in `text`
                will be escaped when rendered as HTML.
        """
        self._text = text

    @property
    def text(self):
        """Get the text content."""
        return self._text

    def _to_html(self, context):
        return escape(self._text)


class HTML(Element):
    """HTML content element."""

    def __init__(self, content):
        """
        Construct the :class:`HTML`.

        Args:
            content (str): The HTML content.  The HTML entities will be
                output as-is when rendered as HTML.
        """
        self._content = content

    @property
    def content(self):
        """Get the HTML content."""
        return self._content

    def _to_html(self, context):
        return self._content


class MagicHTML(Element):
    """Wrapping object with `__html__` method into :class:`Element`."""

    def __init__(self, obj, cacheable=False):
        """
        Construct the :class:`MagicHTML`.

        Args:
            obj: The object to be wrapped.
            cacheable (bool): Whether or not to cache the output of `__html__`
                method?  (default :obj:`False`)
        """
        self._obj = obj
        self._cacheable = cacheable
        self._cached = None

    @property
    def obj(self):
        """Get the wrapped object."""
        return self._obj

    @property
    def cacheable(self):
        """Whether or not the output of `__html__` is cacheable?"""
        return self._cacheable

    def _to_html(self, context):
        if self.cacheable:
            if self._cached is None:
                self._cached = self.obj.__html__()
            return self._cached
        else:
            return self.obj.__html__()


class MathEquation(Element):
    """Base class for :class:`InlineMath` and :class:`BlockMath`."""

    def __init__(self, mathjax):
        """
        Construct the :class:`MathEquation`.

        Args:
            mathjax (str): The source of MathJax equation.  Must not be empty.

        Raises:
            ValueError: If `mathjax` is empty.
        """
        if not mathjax:
            raise ValueError('`mathjax` must not be empty')
        self._mathjax = str(mathjax)

    @property
    def mathjax(self):
        """Get the MathJax equation."""
        return self._mathjax

    def get_libraries(self):
        return [MathJax()]


class InlineMath(MathEquation):
    """Inline math equation."""

    def _to_html(self, context):
        content = escape(self._mathjax)
        return '<span class="inline-math">\\(' + content + '\\)</span>'


class BlockMath(MathEquation):
    """Block math equation."""

    def _to_html(self, context):
        content = escape(self._mathjax)
        return '<div class="block-math">$$' + content + '$$</div>'


class Resource(Element):
    """
    Base class for resource elements.

    Resources are particular types of binary data, for example, images
    and CSV files.  This class provides utilities to embed such resources
    in a report, as HTML content.
    """

    # list of (mime_type, [extensions...]).
    _mime_types = [
        ('image/png', ['.png']),
        ('image/jpeg', ['.jpg', '.jpeg']),
        ('image/bmp', ['.bmp']),
        ('text/csv', ['.csv']),
        ('text/plain', ['.txt']),
        ('text/html', ['.htm', '.html']),
        ('application/octet-stream', ['.bin']),
    ]

    def __init__(self, data, mime_type, title=None):
        """
        Construct the :class:`Resource`.

        Args:
            data (bytes): Binary data of the resource.
            mime_type (str): Mime-type of the resource.
            title (str or None): Title of the resource. (default :obj:`None`)
        """
        if not isinstance(data, six.binary_type):
            raise TypeError('`data` must be binary')
        self._data = data
        self._mime_type = mime_type
        self._title = str(title) if title else None

    @classmethod
    def guess_extension(cls, mime_type):
        """
        Get the preferred file extension for specified mime-type.

        Return value is a string giving a filename extension, including the
        leading dot ".".

        Args:
            mime_type (str): The mime-type.

        Returns:
            str or None: The extension, or :obj:`None` if `mime_type`
                is not registered.
        """
        mime_type = str(mime_type).lower()
        for typ_, extensions in cls._mime_types:
            if typ_ == mime_type:
                return extensions[0]
        return mimetypes.guess_extension(mime_type)

    @classmethod
    def guess_mime_type(cls, extension):
        """
        Get the preferred mime-type for specified file extension.

        Args:
            extension (str): The filename extension, including the leading
                dot ".".

        Returns:
            str or None: The mime-type, or :obj:`None` if `extension`
                is not registered.
        """
        extension = str(extension).lower()
        if not extension.startswith('.'):
            raise ValueError('`extension` must start with "."')
        for typ_, extensions in cls._mime_types:
            if extension in extensions:
                return typ_
        return mimetypes.guess_type('name' + extension)[0]

    @property
    def data(self):
        """Get the binary data of the resource."""
        return self._data

    @property
    def mime_type(self):
        """Get the mime-type of the resource."""
        return self._mime_type

    @property
    def title(self):
        """
        Get the title of the resource.

        Returns:
            str or None: The title, or :obj:`None` if no title is specified.
        """
        return self._title

    def get_uri(self, context):
        """
        Get the URI for embedding the resource into HTML.

        Args:
            context (ToHtmlContext): The context for rendering HTML.

        Returns:
            str: The URI string.
        """
        return (
            'data:' + self.mime_type + ';base64,' +
            base64.b64encode(self.data).decode('utf-8')
        )


class Attachment(Resource):
    """Subclass of :class:`Resource`, rendered as a download link."""

    def _to_html(self, context):
        caption = self.title if self.title else 'Attachment'
        extension = self.guess_extension(self.mime_type) or ''
        return (
            '<div class="attachment block">'
            '<span>{caption}</span>'
            '<a download="{filename}" href="{uri}">Download</a>'
            '</div>'.format(
                caption=escape(caption),
                filename=escape(caption + extension),
                uri=self.get_uri(context)
            )
        )


class Image(Resource):
    """Subclass of :class:`Resource`, rendered as an image."""

    def __init__(self, data, mime_type, title=None, inline=False):
        """
        Construct the :class:`Image`.

        Args:
            data (bytes): Binary data of the image.
            mime_type (str): Mime-type of the image.
            title (str or None): Title of the image. (default :obj:`None`)
            inline (bool): Whether or not to render this image as
                inline HTML element? (default :obj:`False`)
        """
        super(Image, self).__init__(data=data, mime_type=mime_type, title=title)
        self._inline = inline

    @property
    def inline(self):
        """Get whether or not to render this image as inline HTML element."""
        return self._inline

    def _to_html(self, context):
        if self.title:
            title_str = ' title="{}"'.format(escape(self.title))
        else:
            title_str = ''
        if not self.inline:
            classes = ' class="block"'
        else:
            classes = ''
        return '<img src="{src}"{title_str}{classes} />'.format(
            src=escape(self.get_uri(context)),
            title_str=title_str,
            classes=classes,
        )

    @classmethod
    def from_image(cls, image, format='PNG', title=None, inline=False):
        """
        Construct a :class:`Image` from :class:`PIL.Image.Image`.

        Args:
            image (PIL.Image.Image): The PIL image object.
            format (str): The format of the image to be saved as.
                One of {"PNG", "JPG", "JPEG", "BMP"}.
            title (str or None): Title of the image. (default :obj:`None`)
            inline (bool): Whether or not to render this image as
                inline HTML element? (default :obj:`False`)

        Returns:
            Image: The :class:`Image` object.

        Raises:
            RuntimeError: If PIL is not installed.
            TypeError: If `image` is not a :class:`PIL.Image.Image`.
            ValueError: If `format` is not supported.
        """
        if PILImage is None:  # pragma: no cover
            raise RuntimeError('`PIL` is not installed')
        if not isinstance(image, PILImage.Image):
            raise TypeError('`image` must be an instance of `PIL.Image`')
        if format.upper() not in ('PNG', 'JPG', 'JPEG', 'BMP'):
            raise ValueError('Image format {!r} is not allowed: only "PNG", '
                             '"JPG", "JPEG" or "BMP" are supported'.
                             format(format))

        with BytesIO() as f:
            image.save(f, format=format)
            f.seek(0)
            data = f.read()
        mime_type = cls.guess_mime_type('.' + format)
        return Image(data=data, mime_type=mime_type, title=title, inline=inline)

    @classmethod
    def from_figure(cls, figure, dpi=None, tight_bbox=True, title=None,
                    inline=False):
        """
        Construct a :class:`Image` from :class:`matplotlib.pyplot.Figure`.

        Args:
            figure (matplotlib.pyplot.Figure): The matplotlib figure.
            dpi (int or None): DPI for rendering the figure.
                (default :obj:`None`, use the DPI of `figure`)
            tight_bbox (bool): Whether to have a tight bounding box?
                (default :obj:`True`)
            title (str or None): Title of the image. (default :obj:`None`)
            inline (bool): Whether or not to render this image as
                inline HTML element? (default :obj:`False`)

        Returns:
            Image: The :class:`Image` object, carrying a PNG image of
                rendered figure.

        Raises:
            TypeError: If `figure` is not a :class:`matplotlib.pyplot.Figure`.
        """
        if not isinstance(figure, pyplot.Figure):
            raise TypeError('`figure` must be an instance of '
                            '`matplotlib.pyplot.Figure`')

        kwargs = {'format': 'PNG'}
        if dpi:
            kwargs['dpi'] = dpi
        if tight_bbox:
            kwargs['bbox_inches'] = 'tight'

        with BytesIO() as f:
            figure.savefig(f, **kwargs)
            f.seek(0)
            data = f.read()

        return Image(data=data, mime_type='image/png', title=title,
                     inline=inline)


if PILImage is not None:
    register_as_element(PILImage.Image, Image.from_image)
register_as_element(pyplot.Figure, Image.from_figure)


class DataFrameTable(Element):
    """Element to render :class:`pandas.DataFrame` as an HTML table."""

    def __init__(self, data_frame):
        """
        Construct the :class:`DataFrameTable`.

        Args:
            data_frame (pandas.DataFrame): The data frame object.

        Raises:
            TypeError: If `data_frame` is not a :class:`pandas.DataFrame`.
        """
        if not isinstance(data_frame, pd.DataFrame):
            raise TypeError('`data_frame` must be a pandas `DataFrame`')
        self._data_frame = data_frame

    @property
    def data_frame(self):
        """Get the :class:`~pandas.DataFrame` object."""
        return self._data_frame

    def _to_html(self, context):
        table = re.sub(
            r'^<table[^<>]*>',
            '<table class="dataframe">',
            self.data_frame.to_html()
        )
        return '<div class="block">' + table + '</div>'


register_as_element(pd.DataFrame, DataFrameTable)
