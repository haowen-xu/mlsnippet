import functools
import random
import unittest

import numpy as np
import pytest
import six
from mock import Mock

from mlsnippet.datafs import *


def _to_cont(i):
    s = str(i)
    if six.PY3:
        s = s.encode('utf-8')
    return s


class _DummyDataFS(DataFS):

    def __init__(self):
        super(_DummyDataFS, self).__init__(
            DataFSCapacity.READ_DATA | DataFSCapacity.READ_META)

        self._names = [str(i) for i in range(10)]
        self._files_data = {n: _to_cont(i) for i, n in enumerate(self._names)}
        self._files_meta = {n: {n: 1, 'z': n + ' z'} for n in self._names}
        self._get_meta_tuple = lambda n, meta_keys: \
            tuple(self._files_meta[n].get(k) for k in (meta_keys or ()))

    def clone(self):
        return _DummyDataFS()

    def iter_names(self):
        return iter(self._names)

    def sample_names(self, n_samples):
        return [random.choice(self._names)
                for k in range(min(n_samples, len(self._names)))]

    def get_data(self, filename):
        return self._files_data[filename]

    def get_meta(self, filename, meta_keys):
        return self._get_meta_tuple(filename, meta_keys)

    def _init(self):
        pass

    def _close(self):
        pass


class DataFlowCommonChecks(object):

    def check_common_props_and_methods(self, factory):
        fake_fs = Mock(spec=DataFS, init=Mock(), close=Mock())

        # test default args
        flow = factory(fs=fake_fs, batch_size=256)
        self.assertIs(fake_fs, flow.fs)
        self.assertEquals(256, flow.batch_size)
        self.assertFalse(flow.skip_incomplete)
        self.assertTrue(flow.with_names)
        self.assertIsNone(flow.meta_keys)

        # test custom args
        flow = factory(fs=fake_fs, batch_size=256, with_names=False,
                       meta_keys=iter('abc'), skip_incomplete=True)
        self.assertIs(fake_fs, flow.fs)
        self.assertEquals(256, flow.batch_size)
        self.assertTrue(flow.skip_incomplete)
        self.assertFalse(flow.with_names)
        self.assertEquals(('a', 'b', 'c'), flow.meta_keys)

        # test init and close
        flow = factory(fs=fake_fs, batch_size=256)
        self.assertFalse(fake_fs.init.called)
        self.assertFalse(fake_fs.close.called)
        with flow:
            self.assertTrue(fake_fs.init.called)
            self.assertFalse(fake_fs.close.called)
        self.assertTrue(fake_fs.init.called)
        self.assertTrue(fake_fs.close.called)


class DataFSForwardFlowTestCase(unittest.TestCase, DataFlowCommonChecks):

    def test_common_props_and_methods(self):
        self.check_common_props_and_methods(DataFSForwardFlow)

    def test_iterator(self):
        fs = _DummyDataFS()

        # 1st arg set
        flow = DataFSForwardFlow(fs, 4, with_names=True, meta_keys=None,
                                 skip_incomplete=False)
        self.assertEquals(3, len(list(flow)))
        for i, batch in enumerate(flow):
            self.assertEquals(2, len(batch))
            np.testing.assert_equal(
                [str(i * 4 + j) for j in range(4 if i < 2 else 2)],
                batch[0]
            )
            np.testing.assert_equal(
                [_to_cont(i * 4 + j) for j in range(4 if i < 2 else 2)],
                batch[1]
            )

        # 2nd arg set
        meta_keys = ['z'] + [str(i) for i in range(10)]
        flow = DataFSForwardFlow(fs, 4, with_names=True, meta_keys=meta_keys,
                                 skip_incomplete=True)
        self.assertEquals(2, len(list(flow)))
        for i, batch in enumerate(flow):
            self.assertEquals(13, len(batch))
            np.testing.assert_equal(
                [str(i * 4 + j) for j in range(4)],
                batch[0]
            )
            np.testing.assert_equal(
                [_to_cont(i * 4 + j) for j in range(4)],
                batch[1]
            )
            np.testing.assert_equal(
                [str(i * 4 + j) + ' z' for j in range(4)],
                batch[2]
            )
            for k in range(10):
                np.testing.assert_equal(
                    [1 if (i * 4 + j) == k else None for j in range(4)],
                    batch[3 + k]
                )

        # 3rd arg set
        meta_keys = ['z'] + [str(i) for i in range(10)]
        flow = DataFSForwardFlow(fs, 4, with_names=False, meta_keys=meta_keys,
                                 skip_incomplete=True)
        self.assertEquals(2, len(list(flow)))
        for i, batch in enumerate(flow):
            self.assertEquals(12, len(batch))
            np.testing.assert_equal(
                [_to_cont(i * 4 + j) for j in range(4)],
                batch[0]
            )
            np.testing.assert_equal(
                [str(i * 4 + j) + ' z' for j in range(4)],
                batch[1]
            )
            for k in range(10):
                np.testing.assert_equal(
                    [1 if (i * 4 + j) == k else None for j in range(4)],
                    batch[2 + k]
                )


class DataFSIndexedFlowTestCase(unittest.TestCase, DataFlowCommonChecks):

    def test_common_props_and_methods(self):
        self.check_common_props_and_methods(
            functools.partial(DataFSIndexedFlow, names=[]))

        fake_fs = Mock()
        flow = DataFSIndexedFlow(fake_fs, 256, ['a', 'b', 'c'])
        self.assertFalse(flow.is_shuffled)
        self.assertIsInstance(flow.names, np.ndarray)
        np.testing.assert_equal(['a', 'b', 'c'], flow.names)

    def test_normal_iterator(self):
        fs = _DummyDataFS()
        names = list('034578')

        # 1st arg set
        flow = DataFSIndexedFlow(fs, 4, names, with_names=True, meta_keys=None,
                                 skip_incomplete=False)
        self.assertEquals(2, len(list(flow)))
        for i, batch in enumerate(flow):
            self.assertEquals(2, len(batch))
            np.testing.assert_equal(
                [names[i * 4 + j] for j in range(4 if i < 1 else 2)],
                batch[0]
            )
            np.testing.assert_equal(
                [_to_cont(int(names[i * 4 + j]))
                 for j in range(4 if i < 1 else 2)],
                batch[1]
            )

        # 2nd arg set
        meta_keys = ['z'] + [str(i) for i in range(10)]
        flow = DataFSIndexedFlow(fs, 4, names, with_names=True,
                                 meta_keys=meta_keys, skip_incomplete=True)
        self.assertEquals(1, len(list(flow)))
        for i, batch in enumerate(flow):
            self.assertEquals(13, len(batch))
            np.testing.assert_equal(
                [names[i * 4 + j] for j in range(4)],
                batch[0]
            )
            np.testing.assert_equal(
                [_to_cont(int(names[i * 4 + j])) for j in range(4)],
                batch[1]
            )
            np.testing.assert_equal(
                [names[i * 4 + j] + ' z' for j in range(4)],
                batch[2]
            )
            for k in range(10):
                np.testing.assert_equal(
                    [1 if int(names[i * 4 + j]) == k else None
                     for j in range(4)],
                    batch[3 + k]
                )

        # 3rd arg set
        meta_keys = ['z'] + [str(i) for i in range(10)]
        flow = DataFSIndexedFlow(fs, 4, names, with_names=False,
                                 meta_keys=meta_keys, skip_incomplete=True)
        self.assertEquals(1, len(list(flow)))
        for i, batch in enumerate(flow):
            self.assertEquals(12, len(batch))
            np.testing.assert_equal(
                [_to_cont(int(names[i * 4 + j])) for j in range(4)],
                batch[0]
            )
            np.testing.assert_equal(
                [names[i * 4 + j] + ' z' for j in range(4)],
                batch[1]
            )
            for k in range(10):
                np.testing.assert_equal(
                    [1 if int(names[i * 4 + j]) == k else None
                     for j in range(4)],
                    batch[2 + k]
                )

    def test_shuffle_iterator(self):
        fs = _DummyDataFS()
        names = list('034578')

        # 1st arg set
        flow = DataFSIndexedFlow(fs, 4, names, with_names=True, meta_keys=None,
                                 shuffle=True, skip_incomplete=False)
        flow._shuffled_indices_iterator = \
            Mock(wraps=flow._shuffled_indices_iterator)
        self.assertEquals(2, len(list(flow)))
        self.assertTrue(flow._shuffled_indices_iterator.called)
        meet = {n: 0 for n in names}
        for i, batch in enumerate(flow):
            self.assertEquals(2, len(batch))
            for j in range(4 if i < 1 else 2):
                meet[batch[0][j]] += 1
        self.assertEquals(len(names), sum(meet.values()))
        self.assertEquals(0, sum([v > 1 for v in meet.values()]))

        # 2nd arg set
        flow = DataFSIndexedFlow(fs, 4, names, with_names=True, meta_keys=None,
                                 shuffle=True, skip_incomplete=True)
        self.assertEquals(1, len(list(flow)))
        meet = {n: 0 for n in names}
        for i, batch in enumerate(flow):
            self.assertEquals(2, len(batch))
            for j in range(4):
                meet[batch[0][j]] = True
        self.assertEquals(4, sum(meet.values()))
        self.assertEquals(0, sum([v > 1 for v in meet.values()]))


class DataFSRandomFlowTestCase(unittest.TestCase, DataFlowCommonChecks):

    def test_common_props_and_methods(self):
        self.check_common_props_and_methods(DataFSRandomFlow)

        fake_fs = Mock()
        flow = DataFSRandomFlow(fake_fs, 256)
        self.assertIsNone(flow.batch_count)
        flow = DataFSRandomFlow(fake_fs, 256, batch_count=3)
        self.assertEquals(3, flow.batch_count)

        with pytest.raises(ValueError, match='`batch_count` must be positive'):
            _ = DataFSRandomFlow(fake_fs, 256, batch_count=0)
        with pytest.raises(ValueError, match='`batch_count` must be positive'):
            _ = DataFSRandomFlow(fake_fs, 256, batch_count=-1)

    def test_iterator(self):
        fs = _DummyDataFS()
        names = '0123456789'

        # 1st arg set
        flow = DataFSRandomFlow(fs, 4, with_names=True, meta_keys=None,
                                batch_count=3, skip_incomplete=False)
        self.assertEquals(3, len(list(flow)))
        meet = {n: 0 for n in names}
        for i, batch in enumerate(flow):
            self.assertEquals(2, len(batch))
            for j in range(4):
                meet[batch[0][j]] += 1
        self.assertEquals(12, sum(meet.values()))

        # 2nd arg set, should result in no batches (all are incomplete)
        flow = DataFSRandomFlow(fs, 11, with_names=True, meta_keys=None,
                                batch_count=3, skip_incomplete=True)
        self.assertEquals(0, len(list(flow)))

        # 3rd arg set, should result in infinite batches
        flow = DataFSRandomFlow(fs, 4, with_names=True, meta_keys=None,
                                skip_incomplete=True)
        meet = {n: 0 for n in names}
        counter = 0
        for batch in flow:
            self.assertEquals(2, len(batch))
            for j in range(4):
                meet[batch[0][j]] += 1
            counter += 1
            if counter == 100:
                break
        self.assertEquals(counter, 100)
        self.assertEquals(400, sum(meet.values()))


if __name__ == '__main__':
    unittest.main()
