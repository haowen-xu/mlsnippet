import re

__all__ = ['escape', 'camel_to_underscore']


def escape(s):
    """Escape HTML entities in `s`."""
    return (s.replace('&', '&amp;').
              replace('>', '&gt;').
              replace('<', '&lt;').
              replace("'", '&#39;').
              replace('"', '&#34;'))


def camel_to_underscore(name):
    """Convert a camel-case name to underscore."""
    s1 = re.sub(CAMEL_TO_UNDERSCORE_S1, r'\1_\2', name)
    return re.sub(CAMEL_TO_UNDERSCORE_S2, r'\1_\2', s1).lower()


CAMEL_TO_UNDERSCORE_S1 = re.compile('([^_])([A-Z][a-z]+)')
CAMEL_TO_UNDERSCORE_S2 = re.compile('([a-z0-9])([A-Z])')
