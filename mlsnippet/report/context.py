import contextlib

from .utils import camel_to_underscore

__all__ = ['UniqueNames', 'ToHtmlContext']


class UniqueNames(object):
    """Class for obtaining unique names."""

    def __init__(self):
        """Construct the :class:`UniqueNames`."""
        self._object_to_name = {}
        self._allocated_names = set()

    def uniquify(self, name, obj=None):
        """
        Get a unique name according to the candidate `name` and the `obj`.

        Args:
            name (str): The candidate name.  Suffices ``_[id]`` will be
                appended to this name, in order to obtain a unique name.
            obj (any): If not :obj:`None`, the uniquified name will be
                memorized for this object, such that the next time
                :meth:`uniquify` is called, the same name will be returned
                instead of obtaining a new one. (default :obj:`None`)

        Returns:
            str: The uniquified name, or the memorized name for `obj`.

        Raises:
            ValueError: If `obj` is specified, and `name` does not equal to
                the `name` specified in the last call.
        """
        def try_allocate(fullname):
            if fullname not in self._allocated_names:
                if obj is not None:
                    # add mapping `obj -> fullname` before adding `fullname`
                    # to `allocated_names`, such that if `obj` is not hashable,
                    # this method will terminate immediately without touching
                    # `allocated_names`
                    self._object_to_name[obj] = (name, fullname)
                self._allocated_names.add(fullname)
                return True

        # force converting name into str type
        name = str(name)

        # if obj is already allocated, return its name immediately
        if obj is not None and obj in self._object_to_name:
            old_name, fullname = self._object_to_name[obj]
            if old_name != name:
                raise ValueError('`name` does not equal to the last name '
                                 'for `obj`: {!r} vs {!r}'.
                                 format(name, old_name))
            return fullname

        # otherwise we start to allocate the name
        if try_allocate(name):
            return name
        i = 1
        while True:
            fullname = '{}_{}'.format(name, i)
            if try_allocate(fullname):
                return fullname
            i += 1


class ToHtmlContext(object):
    """
    The context for :meth:`mlsnippet.report.Element.to_html`.

    This class carries the context for :meth:`mlsnippet.report.Element.to_html`.
    It collects the elements along the call-stack (i.e., the elements along
    the path of calling :meth:`Element.to_html`), and provide various utilities
    for producing the HTML document.
    """

    def __init__(self, is_ipython=False):
        """
        Construct the :class:`ToHtmlContext`.

        Args:
            is_ipython (bool): Whether or not the elements are displayed
                in a Jupyter notebook? (default :obj:`False`)
        """
        self._is_ipython = is_ipython
        self._elements = []
        self._section_level = 0
        self._element_names = UniqueNames()

    @property
    def is_ipython(self):
        """Whether or not the elements are displayed in a Jupyter notebook?"""
        return self._is_ipython

    def iter_elements(self):
        """
        Get an iterator of the :class:`~mlsnippet.report.Element` along the
        call-stack.
        """
        return iter(self._elements)

    @property
    def section_level(self):
        """
        Get the opened section level.

        Returns:
            int: The section level counter, starting from 1.
                If no section has been opened, return 0.
        """
        return self._section_level

    def section_title_tag(self):
        """
        Get the tag for section title at current level.

        Returns:
            str: One of {"h3", "h4", "h5", "h6", "h7"}.

        Raises:
            RuntimeError: If no section has been opened.
        """
        if not self.section_level:
            raise RuntimeError('No section has been opened')
        return 'h{}'.format(self.section_level + 2)

    @contextlib.contextmanager
    def push(self, element):
        """
        Push an :class:`mlsnippet.report.Element` onto the call-stack,
        and open a context.

        Args:
            element (mlsnippet.report.Element): The element to be pushed.
                If it is a :class:`mlsnippet.report.Section`, then the
                :attr:`section_level` counter will be increased by one
                within the opened context.

        Yields:
            A context where the `element` is on the call-stack.

        Raises:
            TypeError: If `element` is not :class:`~mlsnippet.report.Element`.
            RuntimeError: If a :class:`~mlsnippet.report.Section` is pushed,
                but the current :attr:`section_level` is already ``>=5``.
        """
        from .element import Element
        from .container import Section
        if not isinstance(element, Element):
            raise TypeError('`element` must be an instance of `Element`')
        try:
            self._elements.append(element)
            if isinstance(element, Section):
                self._section_level += 1
                if self._section_level > 5:
                    raise RuntimeError('Too deep nested sections: at most 5 '
                                       'levels of sections are supported')
            yield element
        finally:
            if isinstance(element, Section):
                self._section_level -= 1
            self._elements.pop()

    def get_unique_id(self, element, suffix=None):
        """
        Get a unique id for `element` within this context.

        The unique id will be generated w.r.t. the class name of `element`.
        If `element` is a :class:`~mlsnippet.report.NamedElement`, then
        its `name` will be considered instead of its class name, provided
        `name` is not :obj:`None`.

        Args:
            element (mlsnippet.report.Element): The element instance.
            suffix (str or None): If not :obj:`None`, it will be appended
                to the class name or `name` of the `element`, when obtaining
                the unique id.  (default :obj:`None`)

        Returns:
            str: The unique id for `element`.
        """
        from .element import NamedElement
        candidate_name = None
        if isinstance(element, NamedElement):
            candidate_name = element.name
        if candidate_name is None:
            candidate_name = camel_to_underscore(element.__class__.__name__)
        if suffix:
            candidate_name += suffix
        return self._element_names.uniquify(candidate_name, obj=element)
