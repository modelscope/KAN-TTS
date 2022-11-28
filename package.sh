pip uninstall kantts
python setup.py sdist bdist_wheel
pip install dist/*.whl
