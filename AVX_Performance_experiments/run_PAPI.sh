#!/bin/bash

gcc -O0 -mavx2 Performance_JSA_2_PAPI.c -I/${PAPI_DIR}/include -L/${PAPI_DIR}/lib -o avx -lpapi

#echo "First a unit Test"
#./avx testing -u -s=80 -f=all

echo "Followed by a test_measurement"
./avx testing -t -f=all
