set -o errexit
rm -rf dist build slidescore_sdk.egg-info
python setup.py sdist bdist_wheel
python -m twine upload --verbose -u __token__ dist/*
pip  uninstall  SlideScore-sdk
pip  install --no-deps --user --no-cache-dir --force-reinstall slidescore-sdk
pause