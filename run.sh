#!/usr/bin/env bash
# running production calculations
# Ilya Balabin <ibalabin@avicenna-bio.com>

export thisdir=`pwd`
export runrf='Focus.py'

for dir in production_??? ; do
    
    cd $dir; echo "==> in $dir"
    ./$runrf &> logfile.log &
    cd $thisdir
    sleep 3

done

exit 0
