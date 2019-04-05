# File structure

# --- datasets
#	  --- images
#		  --- SUN397
#		  --- val_large
#	  --- labels
#	  	--- SUN397_labels
#	  	--- val_large_labels

# --- models
#   --- __init__.py  
# --- results
#	  --- SUN397_results
#	  --- val_large_results




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

echo "Extracting datasets from compressed files..."
tar -xvf SUN397.tar.gz datasets/images
tar -xvf val_large.tar datasets/images

echo "Downloading labels..."
mkdir datasets/labels/SUN397_labels
mkdir datasets/labels/val_large_labels
wget https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip
unzip Partitions.zip -d datasets/labels/SUN397_labels/

wget http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar
tar -C datasets/labels/val_large_labels/ -xvf filelist_places365-standard.tar
