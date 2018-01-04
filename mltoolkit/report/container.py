from .element import Element, Text, as_element, escape

__all__ = ['Container', 'Block', 'Paragraph', 'paragraph_text', 'Section']


class Container(Element):
    """
    :class:`Element` class built up with children elements.

    The :class:`Container` class is a base class for implementing
    :class:`Element` which is rendered mainly by rendering children
    elements one after another.
    """

    def __init__(self, children=None):
        """
        Construct the :class:`Container`.

        Args:
            children: An :class:`Element`, or a list of objects which can
                be converted into :class:`Element` via :func:`as_element`.
                (default :obj:`None`, no children is specified)
        """
        if isinstance(children, Element):
            self._children = [children]
        else:
            self._children = [as_element(c) for c in (children or ())]

    @property
    def children(self):
        """
        Get the list of children elements.

        Returns:
            list[Element]: The children elements.
        """
        return self._children

    def add(self, child):
        """
        Add a child :class:`Element`.

        Args:
            child: Object which can be converted into :class:`Element`
                by :func:`as_element`.
        """
        self._children.append(as_element(child))

    def _to_html(self, context):
        return ''.join(c.to_html(context) for c in self.children)

    def get_libraries(self):
        """
        Get the libraries used by this :class:`Container`, and all its children.

        Returns:
            list[Library]: The list of :class:`Library`.
        """
        ret = []
        memo = set()
        for c in self.children:
            c_libs = c.get_libraries()
            if c_libs:
                for lib in c_libs:
                    if lib not in memo:
                        ret.append(lib)
                        memo.add(lib)
        return ret


class Block(Container):
    """
    Subclass of :class:`Container`, which renders all its children elements
    within of ``<div class="block">...</div>``.
    """

    def _to_html(self, context):
        return (
            '<div class="block">' +
            super(Block, self)._to_html(context) +
            '</div>'
        )


class Paragraph(Container):
    """
    Subclass of :class:`Container`, which renders all its children elements
    within of ``<p>...</p>``
    """

    def _to_html(self, context):
        return '<p>' + super(Paragraph, self)._to_html(context) + '</p>'


def paragraph_text(text):
    """
    Create a :class:`Text` enclosed in a :class:`Paragraph`.

    Args:
        text (str): Text content.

    Returns:
        Paragraph: The :class:`Paragraph`.
    """
    return Paragraph(Text(text))


class Section(Container):
    """
    Subclass of :class:`Container`, which renders all its children elements
    within ``<div class="section">...</div>``, with a title.
    """

    def __init__(self, title, children=None):
        """
        Construct the :class:`Section`.

        Args:
            title (str): Title of this :class:`Section`.
            children: List of objects which can be converted into
                :class:`Element` via :func:`as_element`.
                (default :obj:`None`, no children is specified)
        """
        super(Section, self).__init__(children=children)
        self._title = title

    @property
    def title(self):
        """Get the title of this :class:`Section`."""
        return self._title

    def _to_html(self, context):
        return (
            '<div class="section">'
            '<{tag} class="section-title">{title}</{tag}>'
            '<div class="section-body">{body}</div>'
            '</div>'.format(
                tag=context.section_title_tag(),
                title=escape(self.title),
                body=super(Section, self)._to_html(context)
            )
        )
