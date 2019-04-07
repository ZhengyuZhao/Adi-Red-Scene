mkdir datasets
mkdir results

mkdir datasets/SUN397
mkdir datasets/Places
mkdir datasets/SUN397/images
mkdir datasets/SUN397/labels
mkdir datasets/Places/images
mkdir datasets/Places/labels
mkdir results/intermediate
mkdir results/intermediate/Places
mkdir results/intermediate/SUN397
mkdir results/final

echo "Downloading datasets..."
wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
wget http://data.csail.mit.edu/places/places365/val_large.tar

echo "Decompressing datasets..."
tar -C datasets/SUN397/images/ -xvf SUN397.tar.gz
tar -C datasets/Places/images/ -xvf val_large.tar

echo "Deleting compressed data..."
rm datasets/SUN397/images/SUN397.tar.gz
rm datasets/Places/images/val_large.tar
 

echo "Downloading labels..."
wget https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip
unzip Partitions.zip -d datasets/SUN397/labels/

wget http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar
tar -C datasets/Places/labels/ -xvf filelist_places365-standard.tar
