import unittest

from mlsnippet.report import escape, camel_to_underscore


class EscapeTestCase(unittest.TestCase):

    def test_escape(self):
        self.assertEqual(escape('&><\'"'), '&amp;&gt;&lt;&#39;&#34;')


class CamelToUnderscoreTestCase(unittest.TestCase):
    def assert_convert(self, camel, underscore):
        self.assertEqual(
            camel_to_underscore(camel),
            underscore,
            msg='{!r} should be converted to {!r}'.format(camel, underscore)
        )

    def test_camel_to_underscore(self):
        examples = [
            ('simpleTest', 'simple_test'),
            ('easy', 'easy'),
            ('HTML', 'html'),
            ('simpleXML', 'simple_xml'),
            ('PDFLoad', 'pdf_load'),
            ('startMIDDLELast', 'start_middle_last'),
            ('AString', 'a_string'),
            ('Some4Numbers234', 'some4_numbers234'),
            ('TEST123String', 'test123_string'),
        ]
        for camel, underscore in examples:
            self.assert_convert(camel, underscore)
            self.assert_convert(underscore, underscore)
            self.assert_convert('_{}_'.format(camel),
                                '_{}_'.format(underscore))
            self.assert_convert('_{}_'.format(underscore),
                                '_{}_'.format(underscore))
            self.assert_convert('__{}__'.format(camel),
                                '__{}__'.format(underscore))
            self.assert_convert('__{}__'.format(underscore),
                                '__{}__'.format(underscore))
            self.assert_convert(
                '_'.join([s.capitalize() for s in underscore.split('_')]),
                underscore
            )
            self.assert_convert(
                '_'.join([s.upper() for s in underscore.split('_')]),
                underscore
            )
