#!/bin/bash
ps -ef | grep traffic_gen | awk '{print $2}' | xargs kill -9
ps -ef | grep mvfst | awk '{print $2}' | xargs kill -9
ps -ef | grep mm-* | awk '{print $2}' | xargs kill -9
rm core*
rm -rf _build/deps/pantheon/tmp