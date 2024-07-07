set -o errexit
rm -rf dist build slidescore_sdk.egg-info
python setup.py sdist bdist_wheel
python -m twine upload --repository-url https://test.pypi.org/legacy/ -u slidescore dist/*
pip  uninstall  SlideScore-sdk
pip  install --index-url https://test.pypi.org/simple/ --no-deps --user --no-cache-dir --force-reinstall slidescore-sdk
pause