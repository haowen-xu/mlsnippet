import codecs
import os
import jinja2

from mltoolkit.report.container import Container

__all__ = ['Report']


class Report(object):
    """
    Class for building report.
    """

    def __init__(self, title, children, description=None):
        """
        Construct the :class:`Report`.

        Args:
            title (str): The title of the report.
            children: List of objects which can be converted into
                :class:`Element` via :func:`as_element`.
            description (str or None): Optional description of this report.
                (default :obj:`None`)
        """
        self._title = title
        self._body = Container(children)
        self._description = description

    @property
    def title(self):
        """Get the title report."""
        return self._title

    @property
    def body(self):
        """
        Get the body :class:`Container`, which contains all :class:`Element`
        objects converted from `children`.

        Returns:
            Container: The body container.
        """
        return self._body

    @property
    def description(self):
        """
        Get the description of the report.

        Returns:
            str or None: The description of the report.
        """
        return self._description

    def to_html(self):
        """
        Render this report as HTML.

        Returns:
            str: The rendered HTML.
        """
        template_dir = os.path.join(
            os.path.split(os.path.abspath(__file__))[0], 'templates')
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=template_dir),
            autoescape=True
        )
        template = env.get_template('main.html')
        body = self.body.to_html()
        libraries = self.body.get_libraries()
        styles = ''.join(filter(
            lambda s: s, (lib.get_styles() for lib in libraries or ())))
        scripts = ''.join(filter(
            lambda s: s, (lib.get_scripts() for lib in libraries or ())))
        return template.render(
            title=self.title, description=self._description, body=body,
            styles=styles, scripts=scripts
        )

    def save(self, path):
        """
        Save the rendered HTML as file.

        Args:
            path (str): The path of the HTML file.
        """
        with codecs.open(path, 'wb', 'utf-8') as f:
            f.write(self.to_html())
