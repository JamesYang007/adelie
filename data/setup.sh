#!/bin/bash

wget https://archive.ics.uci.edu/static/public/167/arcene.zip
unzip arcene.zip -d arcene
rm -rf arcene.zip

wget https://archive.ics.uci.edu/static/public/169/dorothea.zip
unzip dorothea.zip -d dorothea
rm -rf dorothea.zip

wget https://archive.ics.uci.edu/static/public/170/gisette.zip
unzip gisette.zip -d gisette
rm -rf gisette.zip