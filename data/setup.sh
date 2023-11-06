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

wget https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
unzip electricityloaddiagrams20112014.zip -d electricity
rm -rf electricityloaddiagrams20112014.zip

wget https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip
unzip gene+expression+cancer+rna+seq.zip -d gene
rm -rf gene+expression+cancer+rna+seq.zip
cd gene
tar -xvf TCGA-PANCAN-HiSeq-801x20531.tar.gz
rm -rf TCGA-PANCAN-HiSeq-801x20531.tar.gz
cd ..

# manually download MNIST