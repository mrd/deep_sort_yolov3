#!/bin/bash

DIR=`dirname $0`
LOGFILE=live-counter.log

cd $DIR

sudo -u pirate ./run-live-counter.sh > $LOGFILE 2>&1

