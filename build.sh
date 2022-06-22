$PYTHON -m build -n -x
$PYTHON -m pip install --no-deps .
if errorlevel 1 exit 1