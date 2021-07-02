# mth

## installation

### Virtualbox

- Ubuntu installieren (20.04 funktionierte)
- Guest additions installieren
  - ``sudo apt install build-essential dkms linux-headers-$(uname -r)``
  - -> Geräte -> Gast erweiterungen anlegen
- ``sudo apt install git``
- ``sudo apt install pip-python3``  oderso ähnlich...
- ``sudo apt-get install python3-venv`` 
- ``sudo apt-get install python3-tk`` -> tinker backend für matplotlib
- ``sudo apt-get install ffmpeg`` -> write mp4
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
  -  Damit es in yade ausgeführt werden kann, muss alles im Basisinterpreter installiert sein
  - ``pip install -e mth`` installiert alle Packages ausser pymesh in den Basisinterpreter
- Yade
  - ``sudo apt install yade``
  - Es wird nicht installiert und ein Link in den Basis-Python-Interpreter erstellt
  - Im Virtual environment kann aktiviert werden, dass die packages vom System Interpreter genutzt werden
  - venv/pyvenv.cfg: ``include-system-site-packages = true``
