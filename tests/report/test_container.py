import unittest

from mock import Mock

from mltoolkit.report import *


class ContainerTestCase(unittest.TestCase):

    def test_empty(self):
        # test empty children
        e = Container()
        self.assertEqual(e.children, [])
        self.assertEqual(e.get_libraries(), [])
        self.assertEqual(e.to_html(), '')

    def test_one_element(self):
        # test add one element only
        child = Mock(spec=Element, to_html=Mock(return_value='e:mock'))
        e = Container(child)
        self.assertEqual(len(e.children), 1)
        self.assertIs(e.children[0], child)
        self.assertEqual(e.to_html(), 'e:mock')

    def test_children(self):

        # test with children
        lib = Mock(spec=Library)
        e1 = Mock(spec=Element, to_html=Mock(return_value='e1:mock,'),
                  get_libraries=Mock(return_value=[MathJax(), lib]))
        e2 = Mock(spec=Element, to_html=Mock(return_value='e2:mock,'),
                  get_libraries=Mock(return_value=[MathJax()]))
        e = Container(iter([e1, e2, 'e3:html,']))
        self.assertEqual(len(e.children), 3)
        self.assertIs(e.children[0], e1)
        self.assertIs(e.children[1], e2)
        self.assertIsInstance(e.children[2], HTML)
        self.assertEqual(e.to_html(), 'e1:mock,e2:mock,e3:html,')
        self.assertEqual(e.get_libraries(), [MathJax(), lib])

        # test add children
        e4 = Mock(spec=Element, to_html=Mock(return_value='e4:mock,'))
        e.add(e4)
        self.assertEqual(len(e.children), 4)
        self.assertIs(e.children[3], e4)
        self.assertEqual(e.to_html(), 'e1:mock,e2:mock,e3:html,e4:mock,')

    def test_block(self):
        e = Block(['html'])
        self.assertEqual(e.to_html(), '<div class="block">html</div>')

    def test_paragraph(self):
        e = Paragraph(['1 & 2'])
        self.assertEqual(e.to_html(), '<p>1 & 2</p>')

    def test_paragraph_text(self):
        e = paragraph_text('1 & 2')
        self.assertIsInstance(e, Paragraph)
        self.assertEqual(len(e.children), 1)
        self.assertIsInstance(e.children[0], Text)
        self.assertEqual(e.children[0].text, '1 & 2')
        self.assertEqual(e.to_html(), '<p>1 &amp; 2</p>')

    def test_section(self):
        e = Section(
            title='1 & 2',
            children=['content1', Section(title='3 & 4', children=['content2'])]
        )
        self.assertEqual(e.title, '1 & 2')
        self.assertEqual(
            e.to_html(),
            '<div class="section"><h3 class="section-title">1 &amp; 2</h3>'
            '<div class="section-body">content1'
            '<div class="section"><h4 class="section-title">3 &amp; 4</h4>'
            '<div class="section-body">content2</div></div>'
            '</div></div>'
        )
