#!/bin/bash
ps -ef | grep tunnel_manager* | awk '{print $2}' | xargs kill -9
ps -ef | grep tunnel* | awk '{print $2}' | xargs kill -9
ps -ef | grep traffic_gen | awk '{print $2}' | xargs kill -9
ps -ef | grep mvfst | awk '{print $2}' | xargs kill -9
ps -ef | grep mm-* | awk '{print $2}' | xargs kill -9
ipcrm --all=shm
ipcs -m
rm core*
rm -rf _build/deps/pantheon/tmp