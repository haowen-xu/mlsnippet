import unittest

from mlsnippet.report.library import *


class LibraryTestCase(unittest.TestCase):

    def test_library(self):
        self.assertIsNone(Library().get_styles())
        self.assertIsNone(Library().get_scripts())

    def test_singleton(self):
        class ChildA(SingletonLibrary):
            pass

        class ChildB(SingletonLibrary):
            pass

        self.assertIs(ChildA(), ChildA())
        self.assertIsNot(ChildA(), ChildB())
        self.assertEqual(ChildA(), ChildA())
        self.assertNotEqual(ChildA(), ChildB())
        self.assertEqual(hash(ChildA()), hash(ChildA()))
        self.assertNotEqual(hash(ChildA()), hash(ChildB()))

        class GrandChildA(ChildA):
            pass

        self.assertIs(GrandChildA(), GrandChildA())
        self.assertEqual(GrandChildA(), GrandChildA())
        self.assertEqual(hash(GrandChildA()), hash(GrandChildA()))
        self.assertIsNot(GrandChildA(), ChildA())
        self.assertNotEqual(GrandChildA(), ChildA())
        self.assertNotEqual(hash(GrandChildA()), hash(ChildA()))

    def test_jquery(self):
        self.assertIs(JQuery(), JQuery())
        self.assertEqual(
            JQuery().get_scripts(),
            '<script type="text/javascript">requirejs(["jquery"]);</script>'
        )

    def test_mathjax(self):
        self.assertIs(MathJax(), MathJax())
        self.assertEqual(
            MathJax().get_scripts(),
            '<script type="text/javascript">requirejs(["mathjax"]);</script>'
        )
