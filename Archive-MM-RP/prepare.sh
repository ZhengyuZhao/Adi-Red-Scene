mkdir datasets
mkdir results
mkdir models

mkdir datasets/images
mkdir datasets/labels
mkdir results/SUN397_results
mkdir results/val_large_results
touch models/__init__.py

echo "Downloading datasets..."
wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
wget http://data.csail.mit.edu/places/places365/val_large.tar

echo "Decompressing datasets..."
tar -C datasets/images/SUN397/ -xvf SUN397.tar.gz
tar -C datasets/images/Places/ -xvf val_large.tar

mkdir datasets/labels/SUN397
mkdir datasets/labels/Places

echo "Downloading labels..."
wget https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip
unzip Partitions.zip -d datasets/labels/SUN397/

wget http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar
tar -C datasets/labels/Places/ -xvf filelist_places365-standard.tar
