"""Download the saved Kaggle version, without launching or rerunning a kernel.

Owner authentication is required because the Kaggle notebook is private.
Install kaggle>=2.2,<3 and sign in using `kaggle auth login` first.
The current CLI output command can query an empty live session; this helper
uses its authenticated SDK to request the committed version's output bundle.
"""
import argparse
import io
from pathlib import Path
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, default=2)
    parser.add_argument('--destination', type=Path,
                        default=Path(__file__).parent / 'results/kaggle/version2')
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug = 'burhanahmadkhanbak'
    request.kernel_slug = 'ai651-task2-autoformer-kaggle'
    request.version_number = args.version
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.download_kernel_output(request)
        response.raise_for_status()
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        if not archive.namelist():
            raise RuntimeError('This version has no saved output files.')
        root = args.destination.resolve()
        for member in archive.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise RuntimeError('Unsafe archive path.')
        root.mkdir(parents=True, exist_ok=True)
        archive.extractall(root)
        print(f'Downloaded {len(archive.namelist())} files to {root}')


if __name__ == '__main__':
    main()
