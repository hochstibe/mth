# mth

## installation

### Mit dem Blender Python Interpreter:
```
cd C:\Program Files\Blender Foundation\Blender 2.91\2.91\python>
# pip sollte bereits in site-packages vorhanden sein --> überprüfen mit ensurepip
bin\python.exe -m ensurepip
# pip fehlt im scripts-ordner
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
bin\python.exe -m ensurepip
# jetzt kann pip genutzt werden
Scripts\pip.exe list
```

### Neues Virtual environment (bpy (blendeerpy) kann nicht genutzt werden)
```
# Create virtual environment
python -m venv venv
```