#!/bin/bash
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
mkdir What3D
wget https://lmb.informatik.uni-freiburg.de/data/what3d/renderings.zip -O What3D/renderings.zip
gdrive_download 1UNh7ySRQsZ8qysz_GB9zP9ruskZLje2B What3D/splits.zip
gdrive_download 1wS5B7k2rKB5JsvNKfjs9aOjv3-FQiOEF What3D/ptcloud_object.npz
gdrive_download 1Rw0S0j-I5bsi44zQtw8J1DsI8JiiuIEY What3D/label.npz
cd What3D
unzip renderings.zip
rm renderings.zip
unzip splits.zip
rm splits.zip
cd ..
mv What3D ../