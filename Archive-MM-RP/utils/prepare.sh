mkdir datasets
mkdir results

mkdir datasets/images
mkdir datasets/labels
mkdir results/intermediate
mkdir results/intermediate/Places
mkdir results/intermediate/SUN397
mkdir results/final

echo "Downloading datasets..."
wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
wget http://data.csail.mit.edu/places/places365/val_large.tar

echo "Decompressing datasets..."
tar -C datasets/images/ -xvf SUN397.tar.gz
tar -C datasets/images/ -xvf val_large.tar
mv -T datasets/images/val_large datasets/images/Places

echo "Deleting compressed data..."
rm SUN397.tar.gz
rm val_large.tar
 
mkdir datasets/labels/SUN397/
mkdir datasets/labels/Places/

echo "Downloading labels..."
wget https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip
unzip Partitions.zip -d datasets/labels/SUN397/

wget http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar
tar -C datasets/labels/Places/ -xvf filelist_places365-standard.tar
