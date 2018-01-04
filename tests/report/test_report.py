import codecs
import os
import re
import unittest

from mock import Mock

from mltoolkit.report import *
from mltoolkit.utils import TemporaryDirectory


class ReportTestCase(unittest.TestCase):

    def test_props(self):
        r = Report('title', ['html'])
        self.assertEqual(r.title, 'title')
        self.assertIsNone(r.description)
        self.assertIsInstance(r.body, Container)
        self.assertEqual(len(r.body.children), 1)
        self.assertIsInstance(r.body.children[0], HTML)
        self.assertEqual(r.body.children[0].content, 'html')

        r = Report('title', ['html'], description='description')
        self.assertEqual(r.description, 'description')

    def test_to_html(self):
        template_dir = os.path.join(
            os.path.split(os.path.abspath(__file__))[0],
            '../../mltoolkit/report/templates'
        )
        main_template_path = os.path.join(template_dir, 'main.html')
        with codecs.open(main_template_path, 'rb', 'utf-8') as f:
            tpl = f.read()

        def render(title, body, description, styles, scripts):
            s = tpl
            desc_pattern = '{% if description %}(.*?){{ description }}(.*?)' \
                           '{% endif %}'
            if description:
                s = re.sub(
                    desc_pattern,
                    lambda m: m.group(1) + description + m.group(2),
                    s
                )
            else:
                s = re.sub(desc_pattern, '', s)
            s = s.replace('{{ title }}', title)
            s = s.replace('{{ body|safe }}', body)
            s = s.replace('{{ styles|safe }}', styles)
            s = s.replace('{{ scripts|safe }}', scripts)
            s = s.replace('{% raw %}', '')
            s = s.replace('{% endraw %}', '')
            return s

        # test without description
        r = Report('1 & 2', ['3 & 4'])
        self.assertEqual(r.to_html(), render(
            title='1 &amp; 2',
            body='3 & 4',
            description=None,
            styles='',
            scripts='',
        ))

        # test with description
        r = Report('1 & 2', ['3 & 4'], description='5 & 6')
        self.assertEqual(r.to_html(), render(
            title='1 &amp; 2',
            body='3 & 4',
            description='5 &amp; 6',
            styles='',
            scripts='',
        ))

        # test with libraries
        r = Report('1 & 2', [
            Mock(
                spec=Element,
                to_html=Mock(return_value='content'),
                get_libraries=Mock(return_value=[
                    Mock(
                        spec=Library,
                        get_styles=Mock(
                            return_value='<style>html{}</style>'),
                        get_scripts=Mock(
                            return_value='<script>alert(0);</script>'),
                    ),
                    MathJax(),
                ])
            )
        ])
        self.assertEqual(r.to_html(), render(
            title='1 &amp; 2',
            body='content',
            description=None,
            styles='<style>html{}</style>',
            scripts='<script>alert(0);</script>' + MathJax().get_scripts(),
        ))

    def test_save(self):
        r = Report('1 & 2', ['3 & 4'], description='5 & 6')
        with TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'index.html')
            r.save(path)
            with codecs.open(path, 'rb', 'utf-8') as f:
                self.assertEqual(f.read(), r.to_html())
