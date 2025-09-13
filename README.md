# KMeans HRM for NeurIPS 2025 Open Polymer Challenge

This code reflects work done for the [NeurIPS 2025 Open Polymer Challenge](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)

## Structural Inspiration
-The goal of this design is a graph encoder model.
-The 'high' module is the KMeans head, which creates center masks and then includes points nearby when processing (updating the edges of the carry)
-The 'low' module is a VGAE, which transforms the internal weights (the nodes of the carry)
-The 'low and high modules are used when producing a solution as opposed to the HRM, which the 'HIGH' weights only are used.
-This work is inspired by Green ONIOM and the work of [Dr. Iyengar](http://www.chem.indiana.edu/faculty/srinivasan-s-iyengar/) although not supervised by him.

## Future work
-This repo is scrabbled together and needs organization
-Using energy calculations could increase performance


This project is a derivative work of Thomas Pugh licensed under the Apache License 2.0.

You may not use this file or any of the contents of https://github.com/tspugh/chem-properties/ except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0


This project incorporates code from the following sources:

- sapientinc/HRM (Apache License 2.0)
  Source: https://github.com/sapientinc/HRM/blob/main

Modifications have been made by Thomas Pugh

This project is licensed under the Apache License 2.0.
See individual files and the THIRD_PARTY_LICENSES.md file for details on third-party components.

