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

wget https://archive.ics.uci.edu/static/public/216/amazon+access+samples.zip
unzip amazon+access+samples.zip -d amazon
rm -rf amazon+access+samples.zip
cd amazon
tar -xvf amzn-anon-access-samples.tgz
rm -rf amzn-anon-access-samples.tgz
cd ..

wget https://archive.ics.uci.edu/static/public/328/greenhouse+gas+observing+network.zip
unzip greenhouse+gas+observing+network.zip -d greenhouse
rm -rf greenhouse+gas+observing+network.zip
