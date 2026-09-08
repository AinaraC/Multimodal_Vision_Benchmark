#!/bin/bash

python3 main.py --detpath ../../data/kittiCorrupt/rain/severity_3/training -ce

python3 main.py --detpath ../../data/kittiCorrupt/fog/severity_1/training -ce
python3 main.py --detpath ../../data/kittiCorrupt/fog/severity_3/training -ce
python3 main.py --detpath ../../data/kittiCorrupt/fog/severity_5/training -ce

python3 main.py --detpath ../../data/kittiCorrupt/darkness/severity_1/training -ce
python3 main.py --detpath ../../data/kittiCorrupt/darkness/severity_3/training -ce
python3 main.py --detpath ../../data/kittiCorrupt/darkness/severity_5/training -ce

python3 main.py --detpath ../../data/kittiCorrupt/sunlight/severity_1/training -ce
python3 main.py --detpath ../../data/kittiCorrupt/sunlight/severity_3/training -ce
python3 main.py --detpath ../../data/kittiCorrupt/sunlight/severity_5/training -ce