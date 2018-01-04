from collections import OrderedDict

import numpy as np
import pandas as pd

from .container import *
from .element import *
from .report import Report

__all__ = ['demo_report']


def get_figure(x, func, label, dpi=120):
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(6, 3), dpi=dpi)
    plt.plot(x, func(x), label=label)
    plt.legend()
    return fig


def demo_report():
    """
    Get a demonstration :class:`Report`.

    Returns:
        Report: The demonstration report.
    """
    return Report(
        title='Demonstration Report',
        description='This report demonstrates all report elements.',
        children=[
            Section(
                title='Basic Elements',
                children=[
                    paragraph_text('This is a paragraph with only text.'),
                    paragraph_text('This is a {} paragraph.'.
                                   format(' '.join(['long'] * 100))),
                    Paragraph([
                        '<code>Text</code> element should escape HTML '
                        'entities in its given text: ',
                        Text('<span color="red">red text</span>'),
                    ]),
                    Paragraph([
                        '<code>HTML</code> element should render '
                        'its given HTML as-is: ',
                        HTML('<span style="color:red">red text</span>'),
                    ]),
                    Paragraph([
                        '<code>InlineMath</code> element should render '
                        'math equation along with other elements: ',
                        InlineMath(r'f(a) = \frac{1}{2\pi i} \oint'
                                   r'\frac{f(z)}{z-a}dz'),
                    ]),
                    Paragraph([
                        '<code>BlockMath</code> element should render '
                        'math equation as a dedicated block: ',
                        BlockMath(r'f(a) = \frac{1}{2\pi i} \oint'
                                  r'\frac{f(z)}{z-a}dz'),
                    ]),
                    Paragraph([
                        '<code>Attachment</code> element should render '
                        'as a block with a download link: ',
                        Attachment(
                            data=b'hello, world!', mime_type='text/plain',
                            title='Hello World'
                        )
                    ]),
                    Paragraph([
                        '<code>Image</code> element should by default '
                        'render as a block element.  Images narrower than '
                        '100% width will be rendered as-is: ',
                        Image.from_figure(get_figure(
                            x=np.linspace(0, np.pi * 4, 1001),
                            func=np.sin,
                            label='sin(x)',
                            dpi=96,
                        )),
                        'while images wider than 100% width will be scaled '
                        'to fit the width: ',
                        Image.from_figure(get_figure(
                            x=np.linspace(0, np.pi * 4, 1001),
                            func=np.cos,
                            label='cos(x)',
                            dpi=300,
                        )),
                        'and finally, this is an inline image: ',
                        Image.from_figure(inline=True, figure=get_figure(
                            x=np.linspace(0, np.pi * 4, 1001),
                            func=lambda x: np.sin(x) * np.cos(x),
                            label='sin(x) * cos(x)',
                            dpi=72,
                        )),
                    ]),
                    Paragraph([
                        '<code>DataFrameTable</code> element should render '
                        'a pandas DataFrame into a block with a full-width '
                        'table: ',
                        DataFrameTable(pd.DataFrame(data=OrderedDict([
                            ('A', np.arange(3)),
                            ('B', np.arange(3) + 10),
                            ('C', np.arange(3) + 20),
                        ]))),
                        'wider tables should be wrapped in a scrollable '
                        'block: ',
                        DataFrameTable(pd.DataFrame(data=OrderedDict([
                            (chr(i + 65), np.arange(3) + 10 * i)
                            for i in range(26)
                        ]))),
                    ])
                ]
            ),
            Section(
                title='Containers',
                children=[
                    Paragraph([
                        '<code>Container</code> element simply render its '
                        'children one after another, without wrapper: ',
                        Container([
                            '<em>child of the Container</em>'
                        ])
                    ]),
                    Paragraph([
                        '<code>Block</code> element should wrap its '
                        'children within a dedicated block: '
                    ]),
                    Block([
                        '<em>child of the Block</em>'
                    ]),
                    Paragraph([
                        '<code>Paragraph</code> element should wrap its '
                        'children within a dedicated paragraph: '
                    ]),
                    Paragraph([
                        '<em>child of the Paragraph</em>'
                    ]),
                    Paragraph([
                        '<code>Section</code> element should wrap its '
                        'children within a dedicated section: '
                    ]),
                    Section('Nested Section', [
                        '<em>child of the Section</em>'
                    ]),
                ]
            )
        ]
    )
