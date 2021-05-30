# mth

## installation

### Yade

```
sudo apt-get install yade
# Yade ausführen:
yade
# Python script mit yade ausführen
yade script.py
```

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

### Virtualbox

- Ubuntu installieren (20.04 funktionierte)
- Guest additions installieren
  - ``sudo apt install build-essential dkms linux-headers-$(uname -r)``
  - -> Geräte -> Gast erweiterungen anlegen
- ``sudo apt install git``
- ``sudo apt install pip-python3``  oderso ähnlich...
- ``sudo apt-get install python3-venv`` 
- ``sudo apt-get install python3-tk`` -> tinker backend für matplotlib
- Check python interpreter location: ``python3; import sys; sys.executable``  
- Pymesh: https://github.com/PyMesh/PyMesh
  - ``git clone https://github.com/PyMesh/PyMesh.git``
  - ``cd PyMesh``
  - ``git submodule update --init``
  - ``export PYMESH_PATH=$(pwd)``
  - ``sudo apt-get install libeigen3-dev libgmp-dev libgmpxx4ldbl libmpfr-dev libboost-dev libboost-thread-dev libtbb-dev python3-dev cmake``
  - ``python3 build.py all`` ($Pymesh)  
  - ``mkdir build`` ($PyMesh)
  - ``cd build``
  - ``cmake ..``
  - ``make`` 
  - ``make tests``
  - ``cd PyMesh``
  - Installieren des PyMesh packages:
    - Basis interpreter ``sudo python3 setup.py install``
    - Venv ``sudo /home/stefan/mth/venv/bin/python setup.py install``
- MTh Github repo
  - ``cd /home/stefan``
  - ``git clone https://github.com/hochstibe/mth.git``
  -  ``python3 -m venv venv``
  - ``source venv/bin/activate``
  - ``venv/bin/pip install -r requirements.txt``
  - ``sudo venv/bin/pip install -e .``
- Yade
  - ``sudo apt install yade``
  - Es wird nicht installiert und ein Link in den Basis-Python-Interpreter erstellt
  - Im Virtual environment kann aktiviert werden, dass die packages vom System Interpreter genutzt werden
  - venv/pyvenv.cfg: ``include-system-site-packages = true``
