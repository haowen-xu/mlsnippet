import subprocess
import uuid
from contextlib import contextmanager

import six

__all__ = ['temporary_mongodb']


@contextmanager
def temporary_mongodb():
    """
    Open a temporary MongoDB server for testing, within context.

    Yields:
        str: The connection string to the server.
    """
    daemon_name = uuid.uuid4().hex
    output = subprocess.check_output([
        'docker', 'run', '--rm', '-d',
        '--name', daemon_name,
        '-e', 'MONGO_INITDB_ROOT_USERNAME=root',
        '-e', 'MONGO_INITDB_ROOT_PASSWORD=123456',
        '-p', '27017:27017',
        'mongo'
    ])
    if not (isinstance(output, six.binary_type) and six.PY2):
        output = output.decode('utf-8')
    print('Docker daemon started: {}'.format(output.strip()))
    try:
        conn_str = 'mongodb://root:123456@127.0.0.1:27017/admin'
        yield conn_str
    finally:
        _ = subprocess.check_output(['docker', 'kill', daemon_name])
