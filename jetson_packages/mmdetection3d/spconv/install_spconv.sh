git clone https://github.com/FindDefinition/cumm
git clone https://github.com/traveller59/spconv

# El orden de instalación en https://github.com/traveller59/spconv
pip3 install -e /cumm 

# Solución sacada de https://github.com/traveller59/spconv/issues/468
# Eliminar cumm de la seccion requires en pyproject.toml 
mv pyproject.toml /spconv/pyproject.toml
# Solución sacada de https://github.com/traveller59/spconv/issues/726
# Eliminar la linea 44 de setup.py
mv setup.py /spconv/setup.py
    
pip3 install -e /spconv
# Para buildear spconv, pero no funciona
python3 -c "import spconv"
