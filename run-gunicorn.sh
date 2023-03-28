#!/bin/bash

gunicorn -w 1 --threads 4 -b 0.0.0.0 -t 100 \
	--access-logfile access.log \
	app3:app
