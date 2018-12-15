import codecs
import contextlib
import os
import re
import unittest
from io import BytesIO

import numpy as np
import six
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from PIL import Image as PILImage
from mock import Mock

from mlsnippet.report import *


class ElementTestCase(unittest.TestCase):

    def test_ipython_display(self):
        e = Element()
        e._to_html = Mock(return_value='xyz')
        s = e._repr_html_()

        css_path = os.path.join(
            os.path.split(os.path.abspath(__file__))[0],
            '../../mlsnippet/report/templates/ipython.css'
        )
        with codecs.open(css_path, 'rb', 'utf-8') as f:
            css = f.read()

        self.assertEqual(
            s,
            '<div class="mlsnippet-element"><style>{css}</style>'
            'xyz</div>'.format(css=css)
        )

    def test_to_html(self):
        def _to_html(ctx):
            self.assertIsInstance(ctx, ToHtmlContext)
            self.assertEqual(list(ctx.iter_elements()), [e])
            return ''

        e = Element()
        e._to_html = Mock(wraps=_to_html)

        # test default context
        e.to_html()
        self.assertTrue(e._to_html.called)

    def test_get_libraries(self):
        self.assertIsNone(Element().get_libraries())


class AsElementTestCase(unittest.TestCase):

    def test_as_element(self):
        class MyClass(object):
            pass

        class MyElement(Element):
            def __init__(self, obj):
                self.obj = obj

        # e: Element -> e
        e = Element()
        self.assertIs(as_element(e), e)

        # s: str -> HTML(s)
        s = 'raw html'
        e = as_element(s)
        self.assertIsInstance(e, HTML)
        self.assertEqual(e.content, s)

        # o: MyClass {__html__: () -> str} -> MagicHTML(o)
        o = MyClass()
        o.__html__ = lambda: 'magic html'
        e = as_element(o)
        self.assertIsInstance(e, MagicHTML)
        self.assertIs(e.obj, o)

        # o: MyClass -> TypeError
        o = MyClass()
        with pytest.raises(
                TypeError, match='Cannot convert a `MyClass` object into '
                                 '`Element`: no conversion function is '
                                 'registered'):
            _ = as_element(o)

        # register the conversion function for MyClass
        register_as_element(MyClass, lambda o: MyElement(o))

        # o: MyClass -> MyElement(o)
        o = MyClass()
        e = as_element(o)
        self.assertIsInstance(e, MyElement)
        self.assertIs(e.obj, o)

        # o: MyClass {__html__: () -> str} -> MyElement(o)
        o = MyClass()
        o.__html__ = lambda: 'magic html'
        e = as_element(o)
        self.assertIsInstance(e, MyElement)
        self.assertIs(e.obj, o)

    def test_register_as_element(self):
        with pytest.raises(
                TypeError, match='`typ_` must be a class'):
            register_as_element(1, lambda x: Element())


class NamedElementTestCase(unittest.TestCase):

    def test_name_error(self):
        self.assertIsNone(NamedElement().name)
        self.assertEqual(NamedElement('my-name').name, 'my-name')

        with pytest.raises(
                ValueError, match='`name` must be a non-empty string matching '
                                  'pattern'):
            _ = NamedElement('')

        with pytest.raises(
                ValueError, match='`name` must be a non-empty string matching '
                                  'pattern'):
            _ = NamedElement('?abc')


class TextElementsTestCase(unittest.TestCase):

    def test_text(self):
        e = Text('1&<>')
        self.assertEqual(e.text, '1&<>')
        self.assertEqual(e.to_html(), '1&amp;&lt;&gt;')

    def test_html(self):
        e = HTML('1&<>')
        self.assertEqual(e.content, '1&<>')
        self.assertEqual(e.to_html(), '1&<>')

    def test_magic_html(self):
        def __html__():
            counter[0] += 1
            return str(counter[0])
        counter = [0]
        obj = Mock(__html__=__html__)

        # test not cached
        e = MagicHTML(obj, cacheable=False)
        self.assertIs(e.obj, obj)
        self.assertFalse(e.cacheable)
        self.assertEqual(e.to_html(), '1')
        self.assertEqual(e.to_html(), '2')

        # test cached
        e = MagicHTML(obj, cacheable=True)
        self.assertTrue(e.cacheable)
        self.assertEqual(e.to_html(), '3')
        self.assertEqual(e.to_html(), '3')

        # test default is not cached
        self.assertFalse(MagicHTML(obj).cacheable)

    def test_inline_math(self):
        e = InlineMath('1 & 2')
        self.assertEqual(e.mathjax, '1 & 2')
        self.assertEqual(e.get_libraries(), [MathJax()])
        self.assertEqual(
            e.to_html(),
            '<span class="inline-math">\\(1 &amp; 2\\)</span>'
        )
        with pytest.raises(ValueError, match='`mathjax` must not be empty'):
            _ = InlineMath('')

    def test_block_math(self):
        e = BlockMath('1 & 2')
        self.assertEqual(e.mathjax, '1 & 2')
        self.assertEqual(e.get_libraries(), [MathJax()])
        self.assertEqual(
            e.to_html(),
            '<div class="block-math">$$1 &amp; 2$$</div>'
        )
        with pytest.raises(ValueError, match='`mathjax` must not be empty'):
            _ = BlockMath('')


class ResourceElementsTestCase(unittest.TestCase):

    def test_guess_extension(self):
        self.assertEqual(Resource.guess_extension('image/png'), '.png')
        self.assertEqual(Resource.guess_extension('image/jpeg'), '.jpg')
        self.assertEqual(Resource.guess_extension('image/bmp'), '.bmp')
        self.assertEqual(Resource.guess_extension('text/csv'), '.csv')
        self.assertEqual(Resource.guess_extension('text/plain'), '.txt')
        self.assertEqual(Resource.guess_extension('text/html'), '.htm')
        self.assertEqual(Resource.guess_extension('application/octet-stream'),
                         '.bin')
        self.assertIsNone(Resource.guess_extension(
            'this-ridiculous-mime-type-should-never-exist/i-am-sure'))

        # test upper case
        self.assertEqual(Resource.guess_extension('IMAGE/PNG'), '.png')

    def test_guess_mime_type(self):
        self.assertEqual(Resource.guess_mime_type('.png'), 'image/png')
        self.assertEqual(Resource.guess_mime_type('.jpg'), 'image/jpeg')
        self.assertEqual(Resource.guess_mime_type('.jpeg'), 'image/jpeg')
        self.assertEqual(Resource.guess_mime_type('.bmp'), 'image/bmp')
        self.assertEqual(Resource.guess_mime_type('.csv'), 'text/csv'),
        self.assertEqual(Resource.guess_mime_type('.txt'), 'text/plain')
        self.assertEqual(Resource.guess_mime_type('.htm'), 'text/html')
        self.assertEqual(Resource.guess_mime_type('.html'), 'text/html')
        self.assertEqual(Resource.guess_mime_type('.bin'),
                         'application/octet-stream')
        self.assertIsNone(Resource.guess_mime_type(
            '.this-ridiculous-extension-should-never-exist-i-am-sure'))

        # test upper case
        self.assertEqual(Resource.guess_mime_type('.PNG'), 'image/png')

        # test error
        with pytest.raises(
                ValueError, match=r'`extension` must start with "."'):
            _ = Resource.guess_mime_type('no-leading-dot')

    def test_resource(self):
        ctx = ToHtmlContext()

        e = Resource(b'123', 'text/plain')
        self.assertEqual(e.data, b'123')
        self.assertEqual(e.mime_type, 'text/plain')
        self.assertIsNone(e.title)
        self.assertEqual(e.get_uri(ctx), 'data:text/plain;base64,MTIz')

        e2 = Resource(b'123', 'text/plain', title='456')
        self.assertEqual(e2.title, '456')
        self.assertEqual(e2.get_uri(ctx), 'data:text/plain;base64,MTIz')

        with pytest.raises(TypeError, match='`data` must be binary'):
            _ = Resource(six.text_type('123'), 'text/plain')

    def test_attachment(self):
        ctx = ToHtmlContext()

        # without title
        e = Attachment(b'123', 'text/plain')
        self.assertEqual(
            e.to_html(ctx),
            '<div class="attachment block"><span>Attachment</span>'
            '<a download="Attachment.txt" href="' + e.get_uri(ctx) +
            '">Download</a></div>'
        )

        # with title
        e = Attachment(b'123', 'text/plain', title='1 & 2')
        self.assertEqual(
            e.to_html(ctx),
            '<div class="attachment block"><span>1 &amp; 2</span>'
            '<a download="1 &amp; 2.txt" href="' + e.get_uri(ctx) +
            '">Download</a></div>'
        )


class ImageTestCase(unittest.TestCase):

    def test_image(self):
        ctx = ToHtmlContext()

        # without title
        e = Image(b'123', 'image/png')
        self.assertFalse(e.inline)
        self.assertEqual(
            e.to_html(ctx),
            '<img src="' + e.get_uri(ctx) + '" class="block" />'
        )

        # with title
        e = Image(b'123', 'image/png', title='1 & 2')
        self.assertFalse(e.inline)
        self.assertEqual(
            e.to_html(ctx),
            '<img src="' + e.get_uri(ctx) + '" title="1 &amp; 2" '
                                            'class="block" />'
        )

        # inline image
        e = Image(b'123', 'image/png', inline=True)
        self.assertTrue(e.inline)
        self.assertEqual(
            e.to_html(ctx),
            '<img src="' + e.get_uri(ctx) + '" />'
        )

    def test_from_image(self):
        @contextlib.contextmanager
        def open_data(data):
            with BytesIO(data) as f:
                yield PILImage.open(f)
        im = PILImage.new('RGB', (1, 1), 'red')

        # test :meth:`from_image` with default format
        e = Image.from_image(im)
        with open_data(e.data) as im2:  # type: PILImage.Image
            self.assertEqual(im2.format, 'PNG')
            self.assertEqual(im2.tobytes(), b'\xff\x00\x00')
        self.assertEqual(e.mime_type, 'image/png')

        # test :meth:`from_image` with 'BMP' format
        e = Image.from_image(im, format='BMP')
        with open_data(e.data) as im2:  # type: PILImage.Image
            self.assertEqual(im2.format, 'BMP')
            self.assertEqual(im2.tobytes(), b'\xff\x00\x00')
        self.assertEqual(e.mime_type, 'image/bmp')

        # test :meth:`from_image` with title
        e = Image.from_image(im, title='456')
        self.assertEqual(e.title, '456')

        # test error image type
        with pytest.raises(
                TypeError, match='`image` must be an instance of '
                                 '`PIL.Image`'):
            _ = Image.from_image(1)

        # test error format
        with pytest.raises(
                ValueError, match='Image format \'BIN\' is not allowed'):
            _ = Image.from_image(im, 'BIN')

        # test as_element
        e = as_element(im)
        self.assertIsInstance(e, Image)
        with open_data(e.data) as im2:  # type: PILImage.Image
            self.assertEqual(im2.format, 'PNG')
            self.assertEqual(im2.tobytes(), b'\xff\x00\x00')

    def test_from_figure(self):
        @contextlib.contextmanager
        def open_data(data):
            with BytesIO(data) as f:
                yield PILImage.open(f)

        # create the test figure
        fig = plt.figure(dpi=120)
        x = np.arange(0, np.pi * 2, 1001)
        y = np.sin(x)
        plt.plot(x, y)

        # test default settings
        e = Image.from_figure(fig, title='figure title')
        self.assertEqual(e.mime_type, 'image/png')
        self.assertEqual(e.title, 'figure title')
        with open_data(e.data) as im:
            self.assertEqual(im.format, 'PNG')
            default_size = im.size

        # test override dpi
        e = Image.from_figure(fig, dpi=300)
        with open_data(e.data) as im:
            self.assertNotEqual(im.size, default_size)

        # test override bbox-inches
        e = Image.from_figure(fig, tight_bbox=False)
        with open_data(e.data) as im:
            self.assertNotEqual(im.size, default_size)

        # test type error
        with pytest.raises(
                TypeError, match='`figure` must be an instance of '
                                 '`matplotlib.pyplot.Figure`'):
            _ = e.from_figure(1)

        # test as_element
        e = as_element(fig)
        self.assertIsInstance(e, Image)


class DataFrameTableTestCase(unittest.TestCase):

    def test_data_frame_table(self):
        df = pd.DataFrame(
            data={'a': np.arange(2) + 10, 'b': np.arange(2) + 20},
            index=np.arange(2)
        )

        e = DataFrameTable(df)
        self.assertIs(e.data_frame, df)
        s = re.sub(r'^<table[^<>]*>', '<table class="dataframe">', df.to_html())
        self.assertEqual(
            e.to_html(),
            '<div class="block">' + s + '</div>'
        )

        # test type error
        with pytest.raises(
                TypeError, match='`data_frame` must be a pandas `DataFrame`'):
            _ = DataFrameTable(1)
