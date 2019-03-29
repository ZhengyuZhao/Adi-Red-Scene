# File structure of Adi-Red-Scene



# --- data_clean.py
#
# --- data
#	--- datasets
#		--- SUN397
#		--- val_large
#
#	--- labels
#		--- SUN397_labels
#		--- val_large_labels




# mkdir RP_Adi_Red
# cd RP_Adi_Red
mkdir data
cd data
mkdir datasets
mkdir labels
cd datasets

echo "Downloading datasets..."
wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
wget http://data.csail.mit.edu/places/places365/val_large.tar

echo "Extracting datasets from compressed files..."
tar -xvf SUN397.tar.gz
tar -xvf val_large.tar

cd ../labels/
echo "Downloading labels..."
mkdir SUN397_labels
mkdir val_large_labels
wget https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip
unzip Partitions.zip -d ./SUN397_labels/

cd val_large_labels
wget http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar
tar -xvf filelist_places365-standard.tar

echo "Cleaning data..."
cd ../../../
python data_clean.py
