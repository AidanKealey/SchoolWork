#!/bin/bash
#echo $#
if [ ! "$#" -eq "1" ]
then
   echo "Usage $0 filename"
   exit 1
fi

file=`basename "$1" .csv`
echo output will be in "$file"Sorted.csv
echo "time,icao24,lat,lon,velocity,heading,vertrate,callsign,onground,alert,spi,squawk,baroaltitude,geoaltitude,lastposupdate,lastcontact" > "$file"Sorted.csv

sort -o /tmp/$$.data "$1"
cat /tmp/$$.data >> "$file"Sorted.csv 
rm /tmp/$$.data

