from contextlib import contextmanager
from io import BytesIO

import pytest
import six
from mock import Mock

from mlsnippet.datafs import *
from mlsnippet.utils import maybe_close


class StandardFSChecks(object):
    """
    Standard tests for :class:`DataFS` subclasses.

    To actually implement tests for a particular :class:`DataFS` subclass,
    inherit both the :class:`unittest.TestCase` and this class, and provide
    necessary utilities to support the checks.
    """

    def get_snapshot(self, fs):
        """
        Get the snapshot of the specified `fs`.

        Args:
            fs: The fs, from where to collect the snapshot.

        Returns:
            The snapshot.
        """
        raise NotImplementedError()

    @contextmanager
    def temporary_fs(self, snapshot=None, **kwargs):
        """
        Initialize a temporary :class:`DataFS` with specified snapshot,
        and returns a context on this fs.

        Args:
            snapshot: The initial contents of the constructed fs.
            \**kwargs: Additional named arguments passed to the fs constructor.

        Yields:
            DataFS: The constructed fs.
        """
        raise NotImplementedError()

    def run_standard_checks(self, capacity):
        """
        Run all the standard :class:`DataFS` tests.

        Args:
            *capacity: The expected capacity flags.
        """
        capacity = DataFSCapacity(capacity)
        self.check_props_and_basic_methods(capacity)
        self.check_auto_close_all(capacity)
        self.check_read(capacity)
        self.check_write(capacity)
        self.check_meta_read(capacity)
        self.check_meta_write(capacity)

    def check_props_and_basic_methods(self, capacity):
        with self.temporary_fs(strict=True) as fs:
            # check strict mode
            self.assertTrue(fs.strict)

            # check capacity
            self.assertEquals(capacity, fs.capacity)

            # check clone
            fs2 = fs.clone()
            self.assertIsInstance(fs2, fs.__class__)
            self.assertTrue(fs2.strict)

    def check_auto_close_all(self, capacity):
        names = ['a/1.txt', 'a/2.htm', 'b/1.md', 'b/2.rst', 'c']
        if six.PY2:
            to_bytes = lambda s: s
        else:
            to_bytes = lambda s: s.encode('utf-8')
        get_content = lambda n: to_bytes(n) + b' content'
        snapshot = {n: (get_content(n),) for n in names}
        close_counter = [0]

        with self.temporary_fs(snapshot) as fs:
            def wrap_close(close_func, throw_error=False):
                @six.wraps(close_func)
                def inner(*args, **kwargs):
                    close_counter[0] += 1
                    if throw_error:
                        raise RuntimeError('this error should be caught')
                    return close_func(*args, **kwargs)
                return inner

            try:
                f1 = fs.open('a/1.txt', 'r')
                f1.close = wrap_close(f1.close)
                f2 = fs.open('a/2.htm', 'r')
                f2.close = wrap_close(f2.close, throw_error=True)
                self.assertEquals(0, close_counter[0])
            except AttributeError:
                # Some python versions disallow us to modify `.close()`
                # of a file object, ignore such errors, and skip the tests.
                close_counter[0] = 2
        self.assertEquals(2, close_counter[0])

    def check_read(self, capacity):
        names = ['a/1.txt', 'a/2.htm', 'b/1.md', 'b/2.rst', 'c']
        if six.PY2:
            to_bytes = lambda s: s
        else:
            to_bytes = lambda s: s.encode('utf-8')
        get_content = lambda n: to_bytes(n) + b' content'
        snapshot = {n: (get_content(n),) for n in names}

        with self.temporary_fs(snapshot) as fs:
            self.assertDictEqual(snapshot, self.get_snapshot(fs))

            # count
            self.assertEquals(5, fs.count())
            if capacity.can_quick_count():
                old_iter_names = fs.iter_names
                fs.iter_names = Mock(wraps=old_iter_names)
                self.assertEquals(5, fs.count())
                self.assertFalse(fs.iter_names.called)
                fs.iter_names = old_iter_names

            # iter, list and sample names
            self.assertListEqual(names, sorted(fs.iter_names()))
            self.assertIsInstance(fs.list_names(), list)
            self.assertListEqual(names, sorted(fs.list_names()))

            if capacity.can_random_sample():
                for repeated in range(10):
                    for k in range(len(names)):
                        samples = fs.sample_names(k)
                        self.assertIsInstance(samples, list)
                        self.assertEquals(k, len(samples))
                        for sample in samples:
                            self.assertIn(sample, names)
            else:
                with pytest.raises(UnsupportedOperation):
                    _ = fs.sample_names(1)

            # iter and sample files
            self.assertListEqual(
                [(n, to_bytes(n) + b' content') for n in names],
                sorted(fs.iter_files())
            )

            if capacity.can_random_sample():
                for repeated in range(10):
                    for k in range(len(names)):
                        samples = fs.sample_files(k)
                        self.assertIsInstance(samples, list)
                        self.assertEquals(k, len(samples))
                        for sample in samples:
                            self.assertIn(sample[0], names)
                            self.assertEquals(get_content(sample[0]), sample[1])
            else:
                with pytest.raises(UnsupportedOperation):
                    _ = fs.sample_names(1)

            # retrieve, get data and open
            for n in names:
                self.assertEquals(get_content(n), fs.retrieve(n))
                self.assertEquals(get_content(n), fs.get_data(n))
                with maybe_close(fs.open(n, 'r')) as f:
                    self.assertEquals(get_content(n), f.read())
                with pytest.raises(DataFileNotExist):
                    _ = fs.retrieve(n + '.invalid')
                with pytest.raises(DataFileNotExist):
                    with maybe_close(fs.open(n + '.invalid', 'r')):
                        pass

            # isfile, batch_isfile
            for n in names:
                self.assertTrue(fs.isfile(n))
                self.assertFalse(fs.isfile(n + '.invalid-ext'))
                n_dir = n.rsplit('/', 1)[0]
                if n_dir != n:
                    self.assertFalse(fs.isfile(n_dir))
            self.assertListEqual([True] * len(names), fs.batch_isfile(names))
            self.assertListEqual(
                [True, False] * len(names),
                fs.batch_isfile(sum(
                    [[n, n + '.invalid-ext'] for n in names], []))
            )

    def check_write(self, capacity):
        if not capacity.can_write_data():
            with self.temporary_fs() as fs:
                with pytest.raises(UnsupportedOperation):
                    _ = fs.put_data('a/1.txt', b'a/1.txt content')
                with pytest.raises(UnsupportedOperation):
                    _ = fs.put_data('b/2.txt', BytesIO(b'b/2.txt content'))
                with pytest.raises(UnsupportedOperation):
                    with maybe_close(fs.open('c/3.txt', 'w')) as f:
                        f.write(b'c/3.txt content')
            return

        with self.temporary_fs() as fs:
            self.assertDictEqual({}, self.get_snapshot(fs))

            # put_data
            fs.put_data('a/1.txt', b'to be overwritten')
            fs.put_data('a/1.txt', b'a/1.txt content')
            fs.put_data('b/2.txt', BytesIO(b'b/2.txt content'))
            with pytest.raises(TypeError, match='`data` must be bytes or a '
                                                'file-like object'):
                fs.put_data('err1.txt', u'')
            with pytest.raises(TypeError, match='`data` must be bytes or a '
                                                'file-like object'):
                fs.put_data('err2.txt', object())

            # open
            with maybe_close(fs.open('c/3.txt', 'w')) as f:
                f.write(b'to be overwritten')
            with maybe_close(fs.open('c/3.txt', 'w')) as f:
                f.write(b'c/3.txt content')
            with pytest.raises(UnsupportedOperation,
                               match='Invalid open mode'):
                with maybe_close(
                        fs.open('d/4.txt', 'not-a-possible-write-mode')) as f:
                    f.write(b'')

            self.assertDictEqual(
                {
                    'a/1.txt': (b'a/1.txt content',),
                    'b/2.txt': (b'b/2.txt content',),
                    'c/3.txt': (b'c/3.txt content',)
                },
                self.get_snapshot(fs)
            )

    def check_meta_read(self, capacity):
        names = ['a/1.txt', 'a/2.htm', 'b/1.md', 'b/2.rst', 'c']
        if six.PY2:
            to_bytes = lambda s: s
        else:
            to_bytes = lambda s: s.encode('utf-8')
        get_content = lambda n: to_bytes(n) + b' content'
        snapshot = {n: (get_content(n), {'z': n + ' z', n[0]: 1})
                    for n in names}
        meta_keys_iter = lambda: iter('zabc')
        get_meta_values = lambda name: (name + ' z',) + tuple(
            1 if n == name[0] else None for n in 'abc')

        with self.temporary_fs(snapshot) as fs:
            # list_meta
            if capacity.can_list_meta():
                for name in names:
                    self.assertIsInstance(fs.list_meta(name), tuple)
                    self.assertEquals(
                        [name[0], 'z'],
                        sorted(fs.list_meta(name))
                    )
                    with pytest.raises(DataFileNotExist):
                        _ = fs.list_meta(name + '.invalid')
            else:
                for name in names:
                    with pytest.raises(UnsupportedOperation):
                        _ = fs.list_meta(name)

            # get_meta
            if capacity.can_read_meta():
                for name in names:
                    self.assertEquals(
                        get_meta_values(name),
                        fs.get_meta(name, meta_keys_iter())
                    )
                    with pytest.raises(DataFileNotExist):
                        _ = fs.get_meta(name + '.invalid', meta_keys_iter())

                    # empty meta keys
                    self.assertEquals((), fs.get_meta(name, ()))
                    self.assertEquals((), fs.get_meta(name, None))
            else:
                for name in names:
                    with pytest.raises(UnsupportedOperation):
                        _ = fs.get_meta(name, meta_keys_iter())

            # batch_get_meta
            if capacity.can_read_meta():
                expected = []
                query = sum([[name, name + '.invalid'] for name in names], [])
                for name in names:
                    expected.append(get_meta_values(name))
                    expected.append(None)
                self.assertListEqual(
                    expected,
                    fs.batch_get_meta(query, meta_keys_iter())
                )

                # empty meta keys
                expected = [(), None] * len(names)
                self.assertListEqual(
                    expected,
                    fs.batch_get_meta(query, ())
                )
                self.assertListEqual(
                    expected,
                    fs.batch_get_meta(query, None)
                )
            else:
                with pytest.raises(UnsupportedOperation):
                    _ = fs.batch_get_meta(names, meta_keys_iter())

            # get_meta_dict
            if capacity.can_read_meta() and capacity.can_list_meta():
                for name in names:
                    self.assertDictEqual(
                        {'z': name + ' z', name[0]: 1},
                        fs.get_meta_dict(name)
                    )
                    with pytest.raises(DataFileNotExist):
                        _ = fs.get_meta_dict(name + '.invalid')
            else:
                for name in names:
                    with pytest.raises(UnsupportedOperation):
                        _ = fs.get_meta_dict(name)

            # retrieve
            for name in names:
                if capacity.can_read_meta():
                    self.assertEqual(
                        (get_content(name),) + get_meta_values(name),
                        fs.retrieve(name, meta_keys_iter())
                    )
                    # empty meta keys
                    self.assertEqual(
                        (get_content(name),),
                        fs.retrieve(name, ())
                    )
                    self.assertEqual(
                        get_content(name),
                        fs.retrieve(name, None)
                    )
                else:
                    with pytest.raises(UnsupportedOperation):
                        fs.retrieve(name, meta_keys_iter())

            # iter_files
            if capacity.can_read_meta():
                expected = [
                    (name, get_content(name)) + get_meta_values(name)
                    for name in names
                ]
                self.assertListEqual(
                    expected,
                    sorted(fs.iter_files(meta_keys_iter()))
                )
                # empty meta keys
                expected = [
                    (name, get_content(name))
                    for name in names
                ]
                self.assertListEqual(expected, sorted(fs.iter_files(())))
                self.assertListEqual(expected, sorted(fs.iter_files(None)))
            else:
                with pytest.raises(UnsupportedOperation):
                    _ = sorted(fs.iter_files(meta_keys_iter()))

            # sample_files
            if capacity.can_random_sample() and capacity.can_read_meta():
                for repeated in range(10):
                    for k in range(len(names)):
                        sampled = fs.sample_files(k, meta_keys_iter())
                        self.assertIsInstance(sampled, list)
                        self.assertEquals(k, len(sampled))
                        for name, content, z, a, b, c in sampled:
                            self.assertIn(name, names)
                            self.assertEqual(get_content(name), content)
                            self.assertEquals(
                                get_meta_values(name), (z, a, b, c))
                        # empty meta keys
                        sampled = fs.sample_files(k, ())
                        self.assertEquals(k, len(sampled))
                        for name, content in sampled:
                            self.assertIn(name, names)
                            self.assertEqual(get_content(name), content)
                        sampled = fs.sample_files(k, None)
                        self.assertEquals(k, len(sampled))
                        for name, content in sampled:
                            self.assertIn(name, names)
                            self.assertEqual(get_content(name), content)
            else:
                with pytest.raises(UnsupportedOperation):
                    _ = fs.sample_files(len(names), meta_keys_iter())
                    _ = fs.sample_files(len(names), ())
                    _ = fs.sample_files(len(names), None)

        # test strict read meta
        if capacity.can_read_meta():
            with self.temporary_fs(snapshot, strict=True) as fs:
                self.assertTrue(fs.strict)
                for name in names:
                    with pytest.raises(MetaKeyNotExist):
                        _ = fs.get_meta(name, meta_keys_iter())
                    self.assertEquals(
                        (name + ' z', 1),
                        fs.get_meta(name, ['z', name[0]])
                    )

    def check_meta_write(self, capacity):
        names = ['a/1.txt', 'b/2.rst', 'c']
        if six.PY2:
            to_bytes = lambda s: s
        else:
            to_bytes = lambda s: s.encode('utf-8')
        get_content = lambda n: to_bytes(n) + b' content'
        snapshot = {n: (get_content(n),) for n in names}
        get_meta_dict = lambda name: {'z': name + ' z', name[0]: 1}

        if capacity.can_write_meta():
            with self.temporary_fs(snapshot) as fs:
                # put_meta
                fs.put_meta('a/1.txt', {'z': 'a/1.txt z'})
                fs.put_meta('a/1.txt', a=1)
                fs.put_meta('b/2.rst', {'z': 'b/2.rst z'}, b=1)
                fs.put_meta('c', z='c z', c=1)
                with pytest.raises(DataFileNotExist):
                    _ = fs.put_meta('d.invalid', {'z': 'd z'}, d=1)
                self.assertDictEqual(
                    {name: (get_content(name), get_meta_dict(name))
                     for name in names},
                    self.get_snapshot(fs)
                )

                # clear_meta
                for name in names:
                    try:
                        fs.clear_meta(name)
                        self.assertEquals(
                            (get_content(name),),
                            self.get_snapshot(fs)[name]
                        )
                        with pytest.raises(DataFileNotExist):
                            fs.clear_meta(name + '.invalid')
                    except UnsupportedOperation:
                        # clear_meta might require ``LIST_META`` capacity
                        if capacity.can_list_meta():
                            raise

                # clear_and_put_meta
                for name in names:
                    try:
                        fs.clear_and_put_meta(name, {'z': name + ' z'})
                        fs.clear_and_put_meta(name, {'null': 0}, **{name[0]: 1})
                        self.assertEquals(
                            (get_content(name), {name[0]: 1, 'null': 0}),
                            self.get_snapshot(fs)[name]
                        )
                        with pytest.raises(DataFileNotExist):
                            fs.clear_and_put_meta(name + '.invalid')
                    except UnsupportedOperation:
                        # clear_meta might require ``LIST_META`` capacity
                        if capacity.can_list_meta():
                            raise

                for name in names:
                    try:
                        fs.clear_and_put_meta(
                            name, {'z': name + ' z'}, **{name[0]: 1})
                        self.assertEquals(
                            (get_content(name), get_meta_dict(name)),
                            self.get_snapshot(fs)[name]
                        )
                        fs.clear_and_put_meta(name)
                        self.assertEquals(
                            (get_content(name),),
                            self.get_snapshot(fs)[name]
                        )
                    except UnsupportedOperation:
                        # clear_meta might require ``LIST_META`` capacity
                        if capacity.can_list_meta():
                            raise

        else:
            with self.temporary_fs(snapshot) as fs:
                for name in names:
                    with pytest.raises(UnsupportedOperation):
                        _ = fs.put_meta(name, {'z': name}, **{name[0]: 1})
                    with pytest.raises(UnsupportedOperation):
                        fs.clear_meta(name)
                    with pytest.raises(UnsupportedOperation):
                        _ = fs.clear_and_put_meta(
                            name, {'z': name}, **{name[0]: 1})
