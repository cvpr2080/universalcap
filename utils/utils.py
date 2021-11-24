import sys
import os
import shutil
import warnings
import requests
import pidfile
from contextlib import contextmanager
from time import sleep

@contextmanager
def exclusive(pidname):
    done = False
    while not done:
        try:
            with pidfile.PIDFile(pidname):
                yield
                done = True
        except pidfile.AlreadyRunningError:
            sleep(5)


def sync_path(src_path, dst_path, size=None):
    is_dir = os.path.isdir(src_path)
    with exclusive(dst_path+'.pid'):
        if not os.path.exists(dst_path):
            if size is not None and shutil.disk_usage(os.path.dirname(dst_path))[-1] < size:
                warnings.warn('Unable to copy to %s, because it has no enough space.' % dst_path)
                return False
            else:
                try:
                    warnings.warn("Copying to %s" % dst_path)
                    if is_dir:
                        shutil.copytree(src_path, dst_path)
                    else:
                        shutil.copyfile(src_path, dst_path)
                    warnings.warn("Done.")
                    return True
                except:
                    warnings.warn("An error occurred.")
                    if is_dir and os.path.isdir(dst_path):
                        shutil.rmtree(dst_path)
                    if not is_dir and os.path.isfile(dst_path):
                        os.remove(dst_path)
                    return False
        return True


def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


class DummyFile(object):
    def write(self, x): pass

@contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optionala
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner
