// To measure square and unbalanced matrices with -O0 
gcc -mavx2 Performance_JSA.c -o avx -O0

// To measure square and unbalanced matrices with -O3
gcc -mavx2 Performance_JSA.c -o avx -O3

// To measure Layer L91
gcc -mavx2 Performance_JSA_L91.c -o avx

// To measure square matrices with PAPI time measurement library
gcc -mavx2 -O0 Performance_JSA_2_PAPI.c -I/${PAPI_DIR}/include -L/${PAPI_DIR}/lib -o avx -lpapi
