import unittest

import pytest

from mltoolkit.report import *


class UniqueNamesTestCase(unittest.TestCase):

    def test_uniquify(self):
        names = UniqueNames()

        # test no object
        self.assertEqual(names.uniquify('a'), 'a')
        self.assertEqual(names.uniquify('a'), 'a_1')
        self.assertEqual(names.uniquify('a_1'), 'a_1_1')

        # test with object
        obj1, obj2 = object(), object()
        self.assertEqual(names.uniquify('b', obj1), 'b')
        self.assertEqual(names.uniquify('b', obj2), 'b_1')
        self.assertEqual(names.uniquify('b', obj1), 'b')
        self.assertEqual(names.uniquify('b', obj2), 'b_1')
        with pytest.raises(
                ValueError, match='`name` does not equal to the last name '
                                  'for `obj`'):
            _ = names.uniquify('b_1', obj1)

        # test mixing no object and with object
        obj3 = object()
        self.assertEqual(names.uniquify('a', obj3), 'a_2')
        self.assertEqual(names.uniquify('a'), 'a_3')
        self.assertEqual(names.uniquify('a', obj3), 'a_2')
        self.assertEqual(names.uniquify('b'), 'b_2')
        self.assertEqual(names.uniquify('b', obj1), 'b')


class ToHtmlContextTestCase(unittest.TestCase):

    def test_context(self):
        ctx = ToHtmlContext()

        # test empty context
        self.assertEqual(ctx.section_level, 0)
        self.assertListEqual(list(ctx.iter_elements()), [])
        with pytest.raises(RuntimeError, match='No section has been opened'):
            _ = ctx.section_title_tag()

        # test push elements
        element = Element()
        with ctx.push(element) as e:
            self.assertIs(e, element)
            self.assertEqual(ctx.section_level, 0)
            self.assertListEqual(list(ctx.iter_elements()), [e])

        with ctx.push(Section(title='section_1')) as e:
            self.assertEqual(ctx.section_level, 1)
            self.assertListEqual(list(ctx.iter_elements()), [e])

        message = 'Too deep nested sections: at most 5 ' \
                  'levels of sections are supported'
        with ctx.push(Section(title='1')) as s1:
            self.assertEqual(ctx.section_level, 1)
            self.assertEqual(ctx.section_title_tag(), 'h3')
            self.assertEqual(list(ctx.iter_elements()), [s1])
            with ctx.push(Section(title='2')) as s2:
                self.assertEqual(ctx.section_level, 2)
                self.assertEqual(ctx.section_title_tag(), 'h4')
                self.assertEqual(list(ctx.iter_elements()), [s1, s2])
                with ctx.push(Section(title='3')) as s3:
                    self.assertEqual(ctx.section_level, 3)
                    self.assertEqual(ctx.section_title_tag(), 'h5')
                    self.assertEqual(list(ctx.iter_elements()), [s1, s2, s3])
                    with ctx.push(Section(title='4')) as s4:
                        self.assertEqual(ctx.section_level, 4)
                        self.assertEqual(ctx.section_title_tag(), 'h6')
                        self.assertEqual(list(ctx.iter_elements()),
                                         [s1, s2, s3, s4])
                        with ctx.push(Section(title='5')) as s5:
                            self.assertEqual(ctx.section_level, 5)
                            self.assertEqual(ctx.section_title_tag(), 'h7')
                            self.assertEqual(list(ctx.iter_elements()),
                                             [s1, s2, s3, s4, s5])
                            with pytest.raises(RuntimeError, match=message):
                                with ctx.push(Section(title='6')):
                                    pass
                            self.assertEqual(ctx.section_level, 5)
                            self.assertEqual(list(ctx.iter_elements()),
                                             [s1, s2, s3, s4, s5])
                        self.assertEqual(ctx.section_level, 4)
                        self.assertEqual(list(ctx.iter_elements()),
                                         [s1, s2, s3, s4])
                    self.assertEqual(ctx.section_level, 3)
                    self.assertEqual(list(ctx.iter_elements()), [s1, s2, s3])
                self.assertEqual(ctx.section_level, 2)
                self.assertEqual(list(ctx.iter_elements()), [s1, s2])
            self.assertEqual(ctx.section_level, 1)
            self.assertEqual(list(ctx.iter_elements()), [s1])
        self.assertEqual(ctx.section_level, 0)
        self.assertEqual(list(ctx.iter_elements()), [])

        with pytest.raises(
                TypeError, match='`element` must be an instance of `Element`'):
            with ctx.push(object()):
                pass

    def test_unique_id(self):
        ctx = ToHtmlContext()
        self.assertEqual(ctx.get_unique_id(Element()), 'element')
        self.assertEqual(ctx.get_unique_id(Element()), 'element_1')
        e1, e2, e3 = NamedElement(), NamedElement(), NamedElement('e3')
        self.assertEqual(ctx.get_unique_id(e1), 'named_element')
        self.assertEqual(ctx.get_unique_id(e2), 'named_element_1')
        self.assertEqual(ctx.get_unique_id(e3), 'e3')
        self.assertEqual(ctx.get_unique_id(NamedElement('e3')), 'e3_1')
        self.assertEqual(ctx.get_unique_id(e1), 'named_element')
        self.assertEqual(ctx.get_unique_id(e2), 'named_element_1')
        self.assertEqual(ctx.get_unique_id(e3), 'e3')
        self.assertEqual(ctx.get_unique_id(NamedElement('e3')), 'e3_2')
        self.assertEqual(ctx.get_unique_id(NamedElement(), suffix='X'),
                         'named_elementX')
        self.assertEqual(ctx.get_unique_id(NamedElement(), suffix='X'),
                         'named_elementX_1')
        self.assertEqual(ctx.get_unique_id(NamedElement('e3'), suffix='X'),
                         'e3X')
        self.assertEqual(ctx.get_unique_id(NamedElement('e3'), suffix='X'),
                         'e3X_1')
