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
gdrive_download 1-T7tYfQLfMfwnFdaMTnY1EmdTZHF90rJ What3D/ptcloud_0.npz
gdrive_download 1fEIbXDeta7NK2hzlnFYS70s4ZQQ7WpvE What3D/ptcloud_1.npz
gdrive_download 1I2JlfI1qTYO_vigdG_EUSEf4ptcNj6cs What3D/ptcloud_2.npz
gdrive_download 1PzT1sfb9MbJvY8Cm7MMR9WUvUDIZzxV1 What3D/ptcloud_3.npz
gdrive_download 1vWImbLn-t8arL1PkTTho0yhoR9Y9DJZM What3D/ptcloud_4.npz
gdrive_download 1Rw0S0j-I5bsi44zQtw8J1DsI8JiiuIEY What3D/label.npz
gdrive_download 1pDtMJWFMh4h1Lk2VOTRUnRH58VKEMYcD What3D/voxels_object.zip
cd What3D
unzip renderings.zip
rm renderings.zip
unzip splits.zip
rm splits.zip
unzip voxels_object.zip
rm voxels_object.zip
cd ..
mv What3D ../