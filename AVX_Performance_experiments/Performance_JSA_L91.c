/* ==============================================================================================================
* File:		main_02_21.c
* Target: 		Zedboard (Xilinx)
* Compiler: 	ARM V7 gcc compiler
* Authors: 	Javier Fernandez
* 				Irune Agirre
* 				Jon Perez
*  Brief: 		The purpose of this wrapper is to deploy a matrix-matrix multiplication using a safety
*  			matrix-matrix library (Based on Gemm_nn). In the experiment it has been evaluated the following structures:
1) Experiment 1: evaluating all the variables in the internal, intermediate and external loops with the following checksums:
1.1) XOR
1.2) 1's complement
1.3) 2's complement
1.4) Fletcher
1.5) CRC
2) Combinations of XOR, 1's and 2's complement evaluating each variable in the internal loop and fletcher
or CRC in the intermediate loop evaluating the result of the internal loop.
2.1) smm_xor_flet :		internal loop (XOR) and intermediate loop (fletcher)
2.2) smm_xor_crc:	    internal loop (XOR) and intermediate loop (CRC)
2.3) smm_twos_flet: 	internal loop (two's complement) and intermediate loop (fletcher)
2.4) smm_twos_crc :		internal loop (two's complement) and intermediate loop (CRC)
2.5) smm_ones_flet:	  internal loop (one's complement) and intermediate loop (fletcher)
2.6) smm_ones_crc: 		internal loop (one's complement) and intermediate loop (CRC)
2.7) smm_flet_crc: 		internal loop (Fletcher) and intermediate loop (CRC)
3) Last experiments (Not implemented in this ".c")
3.1) Experiment 3a: Evaluate the checksum of each variable in every compute of the multiplication in the
internal loop. In the intermediate loop evaluate the CRC of all variables and the result of the checksum
(result of the internal loop)
3.2) Experiment 3b: Internal loop not modified. In the intermediate loop is calculated the CRC of the result
but not the other variables involved in the multiplication.
3.3) Experiment 3c: Only a part of the variables have been evaluated instead of evaluating all. In the
intermediate loop is calculated the CRC of the result but not the other variables involved in the multiplication.
3.4) Experiment 3d: In the odden values of the internal loop variable are checked all the variables
involved in the multiplication. In all the intermediate computes are evaluated the result of the checksum.
* ============================================================================================================== */


/* ==============================================================================================================
* Name        : main_02_21.c
* Author      : Javier Fdez
* Version     : V23
* Copyright   : IKERLAN
* Date	       : 07/05/2020
* Description : JF has modified the "measure_dc__error_bit" function allowing to insert errors in not rectangular
*         matrix. It has been modified the loop to go through the number of combinations of the matrix A and a new
*         loop to go through the second matrix B has been coded.
==============================================================================================================  */


/* ==============================================================================================================
* 												INCLUDES
* ============================================================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <stdint.h>
#include "assert.h"
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <omp.h>
#include <limits.h>

#if defined __linux__ || defined _WIN32
#include <time.h>
#else
#include "platform.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xtime_l.h"
#include "errno.h"
#endif

#ifdef _WIN32
#include <intrin.h>
#else
#include <emmintrin.h>
#include <immintrin.h>
#endif
/* ==============================================================================================================
* 										DEFINE
* ============================================================================================================== */


/* ==============================================================================================================
* 										MACROS
* ============================================================================================================== */
#if !defined __linux__ && !defined _WIN32
#define DEF_TIME_VAR(t) XTime t = 0;
#define GET_TIME(t) XTime_GetTime(&t)
#ifdef XPAR_CPU_CORTEXR5_0_CPU_CLK_FREQ_HZ
#define CPU_CLK_FREQ_HZ XPAR_CPU_CORTEXR5_0_CPU_CLK_FREQ_HZ ;
#else
#define CPU_CLK_FREQ_HZ XPAR_CPU_CORTEXA53_0_CPU_CLK_FREQ_HZ;
#endif
#define GET_TIME_DIFF(tmr_start, tmr_end, time_interval) if(tmr_end>tmr_start) { time_interval = ((float64_t) tmr_end- (float64_t) tmr_start) / CPU_CLK_FREQ_HZ; } else { time_interval = ((float64_t) tmr_start- (float64_t) tmr_end ) / CPU_CLK_FREQ_HZ; }
#else
#define DEF_TIME_VAR(t) clock_t t;
#define GET_TIME(t) t = clock();
#define GET_TIME_DIFF(tmr_start, tmr_end, f_time_interval) f_time_interval = (float32_t) (fabs(tmr_end - tmr_start) / CLOCKS_PER_SEC)
#endif

/* It has been developed experiments with unbalanced and square dimension matrices. This variables allows to choose the desired option
	 Square_mtrx = TRUE	    => Square matrix
	 Square_mtrx = FALSE  	=> unbalanced_matrix  */
//#define Square_mtrx boolean FALSE;

#define SINGLETABLE_CRC32_UI32(ui32_crc, ui32_data, u) \
    u.ui32 = ui32_data; \
	ui32_crc = kaui32_crc_table[(ui32_crc ^ u.ui8[0u]) & 0x00ffu] ^ (ui32_crc >> 8u); \
	ui32_crc = kaui32_crc_table[(ui32_crc ^ u.ui8[1u]) & 0x00ffu] ^ (ui32_crc >> 8u); \
	ui32_crc = kaui32_crc_table[(ui32_crc ^ u.ui8[2u]) & 0x00ffu] ^ (ui32_crc >> 8u); \
	ui32_crc = kaui32_crc_table[(ui32_crc ^ u.ui8[3u]) & 0x00ffu] ^ (ui32_crc >> 8u); \

	 /* ==============================================================================================================
	 * 										TYPEDEFS
	 * ============================================================================================================== */
	typedef float    float32_t;
typedef double   float64_t;
typedef void     void_t;

typedef union ui32_to_ui8 {
	uint32_t ui32;
	uint8_t ui8[4];
} ui32_to_ui8_t;

typedef union ui64_to_ui32 {
	uint64_t ui64;
	uint32_t ui32[2];
} ui64_to_ui32_t;

typedef union ui32_to_ui16 {
	uint32_t ui32;
	uint16_t ui16[2];
} ui32_to_ui16_t;

typedef union m256i_to_m128i {
	__m256i m512i;
	__m128i m256i[2];
} m256i_to_m128i_t;

typedef enum
{
	TECH_NONE = 0u,

	TECH_DC_OPT,

	TECH_XOR_EXTERNAL,
	TECH_XOR_INTERMEDIATE,
	TECH_XOR_INTERNAL,

	TECH_ONES_EXTERNAL,
	TECH_ONES_INTERMEDIATE,
	TECH_ONES_INTERNAL,

	TECH_TWOS_EXTERNAL,
	TECH_TWOS_INTERMEDIATE,
	TECH_TWOS_INTERNAL,

	TECH_FLETCHER_EXTERNAL,
	TECH_FLETCHER_INTERMEDIATE,
	TECH_FLETCHER_INTERNAL,

	TECH_CRC_EXTERNAL,
	TECH_CRC_INTERMEDIATE,
	TECH_CRC_INTERNAL,

	TECH_XOR_FLET,
	TECH_XOR_CRC,
	TECH_ONES_FLET,
	TECH_ONES_CRC,
	TECH_TWOS_FLET,
	TECH_TWOS_CRC,
	TECH_FLET_CRC,

	TECH_COMB,
	TECH_CRC_INTERMEDIATE_COMB,
	TECH_CRC_INTERNAL_COMB,

	TECH_INTEL_NO_DC,

	TECH_INTEL_XOR_EXTERNAL,
	TECH_INTEL_XOR_INTERMEDIATE,
	TECH_INTEL_XOR_INTERNAL,

	TECH_INTEL_ONES_EXTERNAL,
	TECH_INTEL_ONES_INTERMEDIATE,
	TECH_INTEL_ONES_INTERNAL,

	TECH_INTEL_TWOS_EXTERNAL,
	TECH_INTEL_TWOS_INTERMEDIATE,
	TECH_INTEL_TWOS_INTERNAL,

	TECH_INTEL_FLETCHER_EXTERNAL,
	TECH_INTEL_FLETCHER_INTERMEDIATE,
	TECH_INTEL_FLETCHER_INTERNAL,

	TECH_INTEL_CRC_EXTERNAL,
	TECH_INTEL_CRC_INTERMEDIATE,
	TECH_INTEL_CRC_INTERNAL,
	TECH_INTEL_XOR_FLET,
	TECH_INTEL_XOR_CRC,
	TECH_INTEL_ONES_FLET,
	TECH_INTEL_ONES_CRC,
	TECH_INTEL_TWOS_FLET,
	TECH_INTEL_TWOS_CRC,
	TECH_INTEL_FLET_CRC,

	TECH_INTEL_COMB,
	TECH_MAX
} e_enum_technique;

typedef enum
{
	eSIZE_MIN = 0u,
	eSIZE_20_20,
	eSIZE_40_40,
	eSIZE_80_80,
	eSIZE_160_160,
	eSIZE_320_320,
	eSIZE_640_640,
	eSIZE_MAX
} e_enum_size_2d;

typedef enum
{
	e_FI_VAR_NONE = 0u,
	e_FI_VAR_A,
	e_FI_VAR_B,
	e_FI_VAR_C,
	e_FI_VAR_MAX
} e_fi_var;

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

/* ==============================================================================================================
* 										DEFINES
* ============================================================================================================== */
#define PUT_IN_REGISTER                           /* dummy definition  for Windows 32 */
#define MEASUREMENT_LOOPS       ((uint32_t) 10u)  /*!< Number of measurement test loops */
#define TIME_MEASUREMENT_LOOPS  ((uint32_t) 100u)  /*!< Number of measurements to measure time */

#define MEASUREMENT_LOOPS_DC   ((uint32_t) 1u)  /*!< Number of measurement test loops */

#define MAX_DIM             ((uint32_t) 640u)
#define M                   ((uint32_t) 32u)//128u) /*!< TBD DESCRIPTION */
#define N                   ((uint32_t) 29u)//57600u) /*!< TBD DESCRIPTION */
#define K                   ((uint32_t) 144u)//576u) /*!< TBD DESCRIPTION */

#define INITIAL_REMAINDER 	((uint32_t) 0xFFFFFFFF) /* Initial value of CRC */
/* TESTS */
#define TIME_SEC2USEC       ((uint32_t) 1000000u) /*!< Microseconds per second*/


#ifndef TRUE
#define TRUE  1u
#define FALSE 0u
#endif

typedef uint32_t bool32_t;
/* ==============================================================================================================
* 										CONSTS
* ============================================================================================================== */
static const uint32_t kaui32_crc_table[256u] = /*! CRC Look-Up Table; See: web.mit.edu � freebsd � head � sys � libkern � crc32*/
{
	0x00000000L, 0xF26B8303L, 0xE13B70F7L, 0x1350F3F4L,
	0xC79A971FL, 0x35F1141CL, 0x26A1E7E8L, 0xD4CA64EBL,
	0x8AD958CFL, 0x78B2DBCCL, 0x6BE22838L, 0x9989AB3BL,
	0x4D43CFD0L, 0xBF284CD3L, 0xAC78BF27L, 0x5E133C24L,
	0x105EC76FL, 0xE235446CL, 0xF165B798L, 0x030E349BL,
	0xD7C45070L, 0x25AFD373L, 0x36FF2087L, 0xC494A384L,
	0x9A879FA0L, 0x68EC1CA3L, 0x7BBCEF57L, 0x89D76C54L,
	0x5D1D08BFL, 0xAF768BBCL, 0xBC267848L, 0x4E4DFB4BL,
	0x20BD8EDEL, 0xD2D60DDDL, 0xC186FE29L, 0x33ED7D2AL,
	0xE72719C1L, 0x154C9AC2L, 0x061C6936L, 0xF477EA35L,
	0xAA64D611L, 0x580F5512L, 0x4B5FA6E6L, 0xB93425E5L,
	0x6DFE410EL, 0x9F95C20DL, 0x8CC531F9L, 0x7EAEB2FAL,
	0x30E349B1L, 0xC288CAB2L, 0xD1D83946L, 0x23B3BA45L,
	0xF779DEAEL, 0x05125DADL, 0x1642AE59L, 0xE4292D5AL,
	0xBA3A117EL, 0x4851927DL, 0x5B016189L, 0xA96AE28AL,
	0x7DA08661L, 0x8FCB0562L, 0x9C9BF696L, 0x6EF07595L,
	0x417B1DBCL, 0xB3109EBFL, 0xA0406D4BL, 0x522BEE48L,
	0x86E18AA3L, 0x748A09A0L, 0x67DAFA54L, 0x95B17957L,
	0xCBA24573L, 0x39C9C670L, 0x2A993584L, 0xD8F2B687L,
	0x0C38D26CL, 0xFE53516FL, 0xED03A29BL, 0x1F682198L,
	0x5125DAD3L, 0xA34E59D0L, 0xB01EAA24L, 0x42752927L,
	0x96BF4DCCL, 0x64D4CECFL, 0x77843D3BL, 0x85EFBE38L,
	0xDBFC821CL, 0x2997011FL, 0x3AC7F2EBL, 0xC8AC71E8L,
	0x1C661503L, 0xEE0D9600L, 0xFD5D65F4L, 0x0F36E6F7L,
	0x61C69362L, 0x93AD1061L, 0x80FDE395L, 0x72966096L,
	0xA65C047DL, 0x5437877EL, 0x4767748AL, 0xB50CF789L,
	0xEB1FCBADL, 0x197448AEL, 0x0A24BB5AL, 0xF84F3859L,
	0x2C855CB2L, 0xDEEEDFB1L, 0xCDBE2C45L, 0x3FD5AF46L,
	0x7198540DL, 0x83F3D70EL, 0x90A324FAL, 0x62C8A7F9L,
	0xB602C312L, 0x44694011L, 0x5739B3E5L, 0xA55230E6L,
	0xFB410CC2L, 0x092A8FC1L, 0x1A7A7C35L, 0xE811FF36L,
	0x3CDB9BDDL, 0xCEB018DEL, 0xDDE0EB2AL, 0x2F8B6829L,
	0x82F63B78L, 0x709DB87BL, 0x63CD4B8FL, 0x91A6C88CL,
	0x456CAC67L, 0xB7072F64L, 0xA457DC90L, 0x563C5F93L,
	0x082F63B7L, 0xFA44E0B4L, 0xE9141340L, 0x1B7F9043L,
	0xCFB5F4A8L, 0x3DDE77ABL, 0x2E8E845FL, 0xDCE5075CL,
	0x92A8FC17L, 0x60C37F14L, 0x73938CE0L, 0x81F80FE3L,
	0x55326B08L, 0xA759E80BL, 0xB4091BFFL, 0x466298FCL,
	0x1871A4D8L, 0xEA1A27DBL, 0xF94AD42FL, 0x0B21572CL,
	0xDFEB33C7L, 0x2D80B0C4L, 0x3ED04330L, 0xCCBBC033L,
	0xA24BB5A6L, 0x502036A5L, 0x4370C551L, 0xB11B4652L,
	0x65D122B9L, 0x97BAA1BAL, 0x84EA524EL, 0x7681D14DL,
	0x2892ED69L, 0xDAF96E6AL, 0xC9A99D9EL, 0x3BC21E9DL,
	0xEF087A76L, 0x1D63F975L, 0x0E330A81L, 0xFC588982L,
	0xB21572C9L, 0x407EF1CAL, 0x532E023EL, 0xA145813DL,
	0x758FE5D6L, 0x87E466D5L, 0x94B49521L, 0x66DF1622L,
	0x38CC2A06L, 0xCAA7A905L, 0xD9F75AF1L, 0x2B9CD9F2L,
	0xFF56BD19L, 0x0D3D3E1AL, 0x1E6DCDEEL, 0xEC064EEDL,
	0xC38D26C4L, 0x31E6A5C7L, 0x22B65633L, 0xD0DDD530L,
	0x0417B1DBL, 0xF67C32D8L, 0xE52CC12CL, 0x1747422FL,
	0x49547E0BL, 0xBB3FFD08L, 0xA86F0EFCL, 0x5A048DFFL,
	0x8ECEE914L, 0x7CA56A17L, 0x6FF599E3L, 0x9D9E1AE0L,
	0xD3D3E1ABL, 0x21B862A8L, 0x32E8915CL, 0xC083125FL,
	0x144976B4L, 0xE622F5B7L, 0xF5720643L, 0x07198540L,
	0x590AB964L, 0xAB613A67L, 0xB831C993L, 0x4A5A4A90L,
	0x9E902E7BL, 0x6CFBAD78L, 0x7FAB5E8CL, 0x8DC0DD8FL,
	0xE330A81AL, 0x115B2B19L, 0x020BD8EDL, 0xF0605BEEL,
	0x24AA3F05L, 0xD6C1BC06L, 0xC5914FF2L, 0x37FACCF1L,
	0x69E9F0D5L, 0x9B8273D6L, 0x88D28022L, 0x7AB90321L,
	0xAE7367CAL, 0x5C18E4C9L, 0x4F48173DL, 0xBD23943EL,
	0xF36E6F75L, 0x0105EC76L, 0x12551F82L, 0xE03E9C81L,
	0x34F4F86AL, 0xC69F7B69L, 0xD5CF889DL, 0x27A40B9EL,
	0x79B737BAL, 0x8BDCB4B9L, 0x988C474DL, 0x6AE7C44EL,
	0xBE2DA0A5L, 0x4C4623A6L, 0x5F16D052L, 0xAD7D5351L
};


static const uint32_t kaui32_matrix_size[eSIZE_MAX] =
{
	/* eSIZE_MIN     */ 20,
	/* eSIZE_40_40   */ 40,
	/* eSIZE_80_80   */ 80,
	/* eSIZE_160_160 */ 160,
	/* eSIZE_320_320 */ 320,
	/* eSIZE_640_640 */ MAX_DIM


	/*				 M=32 N=921600 K=27
	*				 M=64 N=230400 K=288
	*				 M=32 N=230400 K=64
	*				 M=128 N=57600 K=576
	*				 M=64 N=57600 K=128
	*				 M=256 N=14400 K=1152
	*				 M=128 N=14400 K=256
	*				 M=512 N=3600 K=2304
	*				 M=256 N=3600 K=512
	*				 M=1024 N=900 K=4608
	*				 M=512 N=900 K=1024
	*				 M=18 N=900 K=1024
	*				 M=256 N=900 K=512
	*				 M=256 N=3600 K=768
	*				 M=18 N=3600 K=512
	*				 M=128 N=3600 K=256
	*				 M=128 N=14400 K=384
	*				 M=18 N=14400 K=256
	*				 M=128 N=14400 K=128
	*				 M=64 N=57600 K=256
	*				 M=18 N=57600 K=128
	*				 M=128 N=57600 K=64
	*				 M=32 N=230400 K=192
	*				 M=18 N=230400 K=64
	*				 */
};

static e_enum_size_2d process_args(int32_t argc, char *argv[], bool32_t *pb32_time_exp, bool32_t *pb32_dc_exp, bool32_t *pb32_ut_exp, bool32_t *pab32_selected_tech);
static bool32_t execute_unit_test(bool32_t ab32_selected_tech[TECH_MAX], float32_t* const paf32_ma, float32_t* const paf32_mb, float32_t* const paf32_mc, float32_t* const paf32_mc_ref);
static int32_t measure_time(e_enum_size_2d e_size_max, bool32_t ab32_selected_tech[TECH_MAX], float32_t* const paf32_ma, float32_t* const paf32_mb, float32_t* const paf32_mc);
static int32_t measure_dc__error_bit_parallelized(e_enum_size_2d e_size_max, bool32_t ab32_selected_tech[TECH_MAX], bool32_t b32_double_error,
	float32_t* const paf32_ma, float32_t* const paf32_mb, float32_t* const paf32_mc,
	float32_t* const paf32_ma_fi, float32_t* const paf32_mb_fi,
	char *argv[]);
static int32_t measure_dc__error_random_values(e_enum_size_2d e_size_max, bool32_t ab32_selected_tech[TECH_MAX], bool32_t b32_consecutive, uint32_t ui32_n_length, uint32_t ui32_n_iterations,
	float32_t* const paf32_ma, float32_t* const paf32_mb, float32_t* const paf32_mc,
	float32_t* const paf32_ma_fi, float32_t* const paf32_mb_fi,
	float32_t* const paf32_ma_rand, float32_t* const paf32_mb_rand, float32_t* const paf32_mc_rand);
static int32_t measure_dc__error_random_value(e_enum_size_2d e_size_max, bool32_t ab32_selected_tech[TECH_MAX],
	float32_t* const paf32_ma, float32_t* const paf32_mb, float32_t* const paf32_mc,
	float32_t* const paf32_ma_fi, float32_t* const paf32_mb_fi,
	float32_t* const paf32_ma_rand, float32_t* const paf32_mb_rand, float32_t* const paf32_mc_rand);
static void_t print_help_commands(const char *pstr_exec_name);

/* ==============================================================================================================
* 										PROTOTYPES OF LOCAL FUNCTIONS
* ============================================================================================================== */
static void_t mem_fi(float32_t* const paf32_m, uint32_t ui32_bit_idx);
static void_t mem_fi_random_value(float32_t* const paf32_m, uint32_t ui32_max_dim, uint32_t ui32_n_length, bool32_t b32_consecutive);
static void_t matrix2zeros(float32_t  * paf32_matrix, uint32_t ui32_max_rows, uint32_t ui32_max_columns);
static void_t matrix2rand(float32_t  * paf32_matrix, uint32_t ui32_max_rows, uint32_t ui32_max_columns);

/* ==============================================================================================================
* 											Experiment 0 : optimization
==============================================================================================================*/
static uint32_t smm_no_dc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_no_dc_opt(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);

/* ==============================================================================================================
* 											Experiment 1 : individual
==============================================================================================================*/
/* 1.1) XOR */
static uint32_t smm_xor_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_xor_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_xor_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
/* 1.2) 1's complement */
static uint32_t smm_ones_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_ones_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_ones_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
/* 1.3) 2's complement */
static uint32_t smm_twos_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_twos_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_twos_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
/* 1.4) Fletcher */
static uint32_t smm_fletcher_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_fletcher_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_fletcher_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static inline uint32_t Fletcher32c_ui32(ui32_to_ui16_t Fletcher, uint32_t ui32_data);
/* 1.5) CRC */
static uint32_t smm_crc_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_crc_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_crc_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static inline uint32_t singletable_crc32c_ui32(uint32_t ui32_crc, uint32_t ui32_data);

/*==============================================================================================================
* 											Experiment 2 : combinations 1.0
==============================================================================================================*/
static uint32_t smm_xor_flet(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_xor_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_twos_flet(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_twos_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_ones_flet(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_ones_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_flet_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);


/*==============================================================================================================
* 											Experiment 3 : combinations 2.0 (JoP)
==============================================================================================================*/
static uint32_t smm_crc_intermediate_comb(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_crc_internal_comb(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_comb(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);

/*==============================================================================================================
* 											Experiment 4 : AVX intel
==============================================================================================================*/

static uint32_t smm_gemm_nn_intrincs_intel(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_xor_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_xor_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_xor_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_twos_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_twos_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_twos_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_ones_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_ones_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_ones_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_fletcher_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_fletcher_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_fletcher_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_crc_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_crc_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_crc_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_xor_flet(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_xor_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_twos_flet(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_twos_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_ones_flet(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_ones_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);
static uint32_t smm_intel_flet_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc);


uint32_t(*ptr_fn_smm_technique[TECH_MAX])(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc) =
{
	smm_no_dc,
	smm_no_dc_opt,

	/* XOR */
	smm_xor_external,
	smm_xor_intermediate,
	smm_xor_internal,

	/* 1's COMPLEMENT */
	smm_ones_external,
	smm_ones_intermediate,
	smm_ones_internal,

	/* 2's COMPLEMENT */
	smm_twos_external,
	smm_twos_intermediate,
	smm_twos_internal,

	/* FLETCHER */
	smm_fletcher_external,
	smm_fletcher_intermediate,
	smm_fletcher_internal,

	/* CRC */
	smm_crc_external,
	smm_crc_intermediate,
	smm_crc_internal,

	/* MIX */
	smm_xor_flet,
	smm_xor_crc,
	smm_ones_flet,
	smm_ones_crc,
	smm_twos_flet,
	smm_twos_crc,
	smm_flet_crc,

	/* ADITIONAL */
	smm_comb,
	smm_crc_intermediate_comb,
	smm_crc_internal_comb

	/* INTRINSIC INTEL INSTRUCTIONS */
	,smm_gemm_nn_intrincs_intel,

	/* XOR */
	smm_intel_xor_external,
	smm_intel_xor_intermediate,
	smm_intel_xor_internal,

	/* 1's COMPLEMENT */
	smm_intel_ones_external,
	smm_intel_ones_intermediate,
	smm_intel_ones_internal,

	/* 2's COMPLEMENT */
	smm_intel_twos_external,
	smm_intel_twos_intermediate,
	smm_intel_twos_internal,

	/* FLETCHER */
	smm_intel_fletcher_external,
	smm_intel_fletcher_intermediate,
	smm_intel_fletcher_internal,

	/* CRC */
	smm_intel_crc_external,
	smm_intel_crc_intermediate,
	smm_intel_crc_internal,

	/* MIX */
	smm_intel_xor_flet,
	smm_intel_xor_crc,
	smm_intel_ones_flet,
	smm_intel_ones_crc,
	smm_intel_twos_flet,
	smm_intel_twos_crc,
	smm_intel_flet_crc,

	/* ADITIONAL */
	smm_comb
};

const char *pstr_technique[TECH_MAX] =
{
	"No DC",
	"No DC opt",

	"XOR_external",
	"XOR_intermediate",
	"XOR_internal",

	"ONES_external",
	"ONES_intermediate",
	"ONES_INTERNAL",

	"TWOS_external",
	"TWOS_intermediate",
	"TWOS_INTERNAL",

	"FLETCHER_external",
	"FLETCHER_intermediate",
	"FLETCHER_INTERNAL",

	"CRC_external",
	"CRC_intermediate",
	"CRC_INTERNAL",

	"XOR_FLET",
	"XOR_CRC",
	"ONES_FLET",
	"ONES_CRC",
	"TWOS_FLET",
	"TWOS_CRC",
	"FLET_CRC",

	"COMB",
	"CRC_INTERMEDIATE_COMB",
	"CRC_INTERNAL_COMB",
	"INTEL_NO_DC",

	"INTEL_XOR_external",
	"INTEL_XOR_intermediate",
	"INTEL_XOR_internal",

	"INTEL_ONES_external",
	"INTEL_ONES_intermediate",
	"INTEL_ONES_INTERNAL",

	"INTEL_TWOS_external",
	"INTEL_TWOS_intermediate",
	"INTEL_TWOS_INTERNAL",

	"INTEL_FLETCHER_external",
	"INTEL_FLETCHER_intermediate",
	"INTEL_FLETCHER_INTERNAL",

	"INTEL_CRC_external",
	"INTEL_CRC_intermediate",
	"INTEL_CRC_INTERNAL",

	"INTEL_XOR_FLET",
	"INTEL_XOR_CRC",
	"INTEL_ONES_FLET",
	"INTEL_ONES_CRC",
	"INTEL_TWOS_FLET",
	"INTEL_TWOS_CRC",
	"INTEL_FLET_CRC",

	"MAXIMUM"
};

#ifdef Square_mtrx
float32_t af32_matrix_a[MAX_DIM * MAX_DIM],
af32_matrix_b[MAX_DIM * MAX_DIM],
af32_matrix_c[MAX_DIM * MAX_DIM],
af32_matrix_c_ref[MAX_DIM * MAX_DIM];

float32_t af32_matrix_a_rand[MAX_DIM * MAX_DIM],
af32_matrix_b_rand[MAX_DIM * MAX_DIM],
af32_matrix_c_rand[MAX_DIM * MAX_DIM];

float32_t af32_matrix_a_fi[MAX_DIM * MAX_DIM],
af32_matrix_b_fi[MAX_DIM * MAX_DIM];
#else
float32_t af32_matrix_a[M * K],
af32_matrix_b[K * N],
af32_matrix_c[M * N],
af32_matrix_c_ref[M * N];

float32_t af32_matrix_a_rand[M * K],
af32_matrix_b_rand[K * N],
af32_matrix_c_rand[M * N];

float32_t af32_matrix_a_fi[M * K],
af32_matrix_b_fi[K * N];
#endif

#if !defined __linux__ || !defined _WIN32
int32_t argc = 4;
const char *argv[] = { "testing", "-t","-s=18", "-f=all" };
#endif


/* ==============================================================================================================
* 	Name:main.c
*
* 	Brief:  Aiming to store each version and for offering the possibility of re-evaluated each experiment it has
* 			been developed this function which will make a call to the version desired to implement or evaluate.
* ============================================================================================================== */

#if defined __linux__ || defined _WIN32
int32_t main(int32_t argc, char *argv[])
#else
int32_t main(void)
#endif
{
	/* Execution configuration */
	bool32_t b32_time_exp = TRUE,
		b32_dc_exp = TRUE,
		b32_ut_exp = TRUE,
		b32_ut_result,
		ab32_selected_tech[TECH_MAX];
	e_enum_size_2d e_size_max;
	uint32_t ui32_fi_max_variables = 10u,
		ui32_fi_iterations = 10u;

	/***********************************************************************************************************************
	*| STEP 0: Remainder of the commands
	***********************************************************************************************************************/
	print_help_commands(argv[0]);

	/***********************************************************************************************************************
	*| STEP 1: Process configuration arguments
	***********************************************************************************************************************/
	printf("\n [1] Process configuration arguments");
	e_size_max = process_args(argc, argv, &b32_time_exp, &b32_dc_exp, &b32_ut_exp, &ab32_selected_tech[0u]);

	/***********************************************************************************************************************
	*| STEP 2: UNIT TESTS
	***********************************************************************************************************************/
	printf("\n\n [2] Execute tests - Unit Tests");
	if (b32_ut_exp)
	{
		b32_ut_result = execute_unit_test(&ab32_selected_tech[0u], &af32_matrix_a[0u], &af32_matrix_b[0u], &af32_matrix_c[0u], &af32_matrix_c_ref[0u]);
		printf("\n  Unit rest result = %4s", b32_ut_result ? "OK" : "FAIL");
	}
	else
	{
		printf("\n ---> User argument requested to skip unit test");
	}

	/***********************************************************************************************************************
	*| STEP 3 - Execute Tests - Measure Time
	***********************************************************************************************************************/
	printf("\n\n [3] Measure Time (number iteration = %u)", MEASUREMENT_LOOPS);
	if (b32_time_exp)
	{
		measure_time(e_size_max, ab32_selected_tech, &af32_matrix_a[0u], &af32_matrix_b[0u], &af32_matrix_c[0u]);
	}
	else
	{
		printf("\n ---> User argument requested to skip time measurements");
	}


	/***********************************************************************************************************************
	*| STEP 4 - Execute Tests - Measure DC
	***********************************************************************************************************************/
	printf("\n\n [4] Measure DC - Execute %u fault injection campaigns with all %u techniques- Iteration with A=B=0", MEASUREMENT_LOOPS_DC, TECH_MAX);
	if (b32_dc_exp)
	{
		/*printf("\n\n\t EXHAUSTIVE - RANDOM VALUE REPLACEMENT ERROR");
		measure_dc__error_random_value(e_size_max, ab32_selected_tech,
		&af32_matrix_a[0u], &af32_matrix_b[0u], &af32_matrix_c[0u],
		&af32_matrix_a_fi[0u], &af32_matrix_b_fi[0u],
		&af32_matrix_a_rand[0u], &af32_matrix_b_rand[0u], &af32_matrix_c_rand[0u]);*/

		printf("\n\n\t EXHAUSTIVE - SINGLE BIT ERROR");
		measure_dc__error_bit_parallelized(e_size_max, ab32_selected_tech, FALSE,
			&af32_matrix_a[0u], &af32_matrix_b[0u], &af32_matrix_c[0u],
			&af32_matrix_a_fi[0u], &af32_matrix_b_fi[0u],
			argv);

		/*printf("\n\n\t EXHAUSTIVE - DOUBLE BIT ERROR");
		measure_dc__error_bit(e_size_max, ab32_selected_tech, TRUE,
		&af32_matrix_a[0u], &af32_matrix_b[0u], &af32_matrix_c[0u],
		&af32_matrix_a_fi[0u], &af32_matrix_b_fi[0u],
		&af32_matrix_a_rand[0u], &af32_matrix_b_rand[0u], &af32_matrix_c_rand[0u]);*/

		/*printf("\n\n\t RANDOM - [1...%u] RANDOM VALUES x %u iterations at random positions", ui32_fi_max_variables, ui32_fi_iterations);
		measure_dc__error_random_values(e_size_max, ab32_selected_tech, FALSE, ui32_fi_max_variables, ui32_fi_iterations,
		&af32_matrix_a[0u], &af32_matrix_b[0u], &af32_matrix_c[0u],
		&af32_matrix_a_fi[0u], &af32_matrix_b_fi[0u],
		&af32_matrix_a_rand[0u], &af32_matrix_b_rand[0u], &af32_matrix_c_rand[0u]);

		//printf("\n\n\t RANDOM - [1...%u] RANDOM VALUES x %u iterations at consecutive positions", ui32_fi_max_variables, ui32_fi_iterations);
		//measure_dc__error_random_values(e_size_max, ab32_selected_tech, TRUE, ui32_fi_max_variables, ui32_fi_iterations,
		&af32_matrix_a[0u], &af32_matrix_b[0u], &af32_matrix_c[0u],
		&af32_matrix_a_fi[0u], &af32_matrix_b_fi[0u],
		&af32_matrix_a_rand[0u], &af32_matrix_b_rand[0u], &af32_matrix_c_rand[0u]); */
	}
	else
	{
		printf("\n ---> User argument requested to skip time measurements");
	}

	printf("\n\n FINISHED \n");

	return 0;
}


/******************************************************************************
** Name:    process_args
******************************************************************************/
/*!
** @brief  Process executable main calling arguments to extract configuration features
**
** @param[in]     argc           Number of arguments
** @param[in]     argv           Array of argument strings
** @param[in,out] pb32_time_exp  Execute time measurement experiments boolean option
** @param[in,out] pb32_dc_exp    Execute DC measurement experiments boolean option
**
** @return  e_enum_size_2d Maximum matrix size
**
******************************************************************************/
static e_enum_size_2d process_args(int32_t argc, char *argv[], bool32_t *pb32_time_exp, bool32_t *pb32_dc_exp, bool32_t *pb32_ut_exp, bool32_t *pab32_selected_tech)
{
	uint32_t ui32_idx,
		ui32_matrix_size;
	char *pstr_arg_size = NULL;
	const char *pstr_arg = NULL;
	e_enum_size_2d e_size = eSIZE_MAX,
		e_size_idx;
	bool32_t b32_time_exp = TRUE,
		b32_dc_exp = TRUE,
		b32_ut_exp = TRUE,
		b32_help = FALSE;
	e_enum_technique e_tech;
	bool32_t ab32_b_tech_all[TECH_MAX],
		ab32_b_tech_best[TECH_MAX],
		ab32_b_tech_best_dc[TECH_MAX],
		ab32_b_tech_best_dc_time[TECH_MAX];

	/* 1. Check input parameters */
	assert(pb32_time_exp != NULL);
	assert(pb32_dc_exp != NULL);

	/* 2. Define technique selections options */
	for (e_tech = TECH_NONE; e_tech < TECH_MAX; e_tech++)
	{
		ab32_b_tech_all[e_tech] = TRUE;
		ab32_b_tech_best[e_tech] = FALSE;
		ab32_b_tech_best_dc[e_tech] = FALSE;
		ab32_b_tech_best_dc_time[e_tech] = FALSE;
	}
	ab32_b_tech_best[TECH_INTEL_NO_DC] = TRUE;

	ab32_b_tech_best_dc[TECH_COMB] = TRUE;
	ab32_b_tech_best_dc[TECH_CRC_INTERMEDIATE_COMB] = TRUE;
	ab32_b_tech_best_dc[TECH_CRC_INTERNAL_COMB] = TRUE;
	ab32_b_tech_best_dc[TECH_ONES_FLET] = TRUE;
	ab32_b_tech_best_dc[TECH_ONES_CRC] = TRUE;
	ab32_b_tech_best_dc[TECH_TWOS_FLET] = TRUE;
	ab32_b_tech_best_dc[TECH_TWOS_CRC] = TRUE;
	ab32_b_tech_best_dc[TECH_FLET_CRC] = TRUE;

	ab32_b_tech_best_dc_time[TECH_INTEL_NO_DC] = TRUE;
	ab32_b_tech_best_dc_time[TECH_INTEL_ONES_INTERNAL] = TRUE;
	ab32_b_tech_best_dc_time[TECH_COMB] = FALSE;
	ab32_b_tech_best_dc_time[TECH_ONES_INTERNAL] = FALSE;
	ab32_b_tech_best_dc_time[TECH_TWOS_INTERNAL] = FALSE;
	ab32_b_tech_best_dc_time[TECH_ONES_FLET] = FALSE;
	ab32_b_tech_best_dc_time[TECH_ONES_CRC] = FALSE;
	ab32_b_tech_best_dc_time[TECH_TWOS_FLET] = FALSE;
	ab32_b_tech_best_dc_time[TECH_CRC_INTERNAL] = FALSE;
	ab32_b_tech_best_dc_time[TECH_TWOS_CRC] = FALSE;

	memcpy(pab32_selected_tech, &ab32_b_tech_all[0], TECH_MAX * sizeof(bool32_t));

	/* 3. If arguments, process them to select the desired execution features*/
	if (argc > 1)
	{
		printf("\n\t Number Of Arguments Passed: %d,", argc);
		printf("\n\t Processing Command Line Arguments Passed----,");

		/* If configuration arguments -> Select which experiment to execute */
		b32_time_exp = FALSE;
		b32_dc_exp = FALSE;
		b32_ut_exp = FALSE;

		for (ui32_idx = 1u; ui32_idx < (uint32_t)argc; ui32_idx++)
		{
			pstr_arg = argv[ui32_idx];
			printf("\n\t\t argv[%d]: %s,", ui32_idx, pstr_arg);
			b32_time_exp = (strcmp(pstr_arg, "-t") == 0) ? TRUE : b32_time_exp;
			b32_ut_exp = (strcmp(pstr_arg, "-u") == 0) ? TRUE : b32_ut_exp;
			b32_dc_exp = (strcmp(pstr_arg, "-dc") == 0) ? TRUE : b32_dc_exp;
			pstr_arg_size = strstr(pstr_arg, "-s=");
			if (pstr_arg_size != NULL)
			{
				pstr_arg_size = &pstr_arg_size[3u];
				if (isdigit(*pstr_arg_size))
				{
					ui32_matrix_size = (uint32_t)strtoul(pstr_arg_size, NULL, 10u);
					for (e_size_idx = eSIZE_MIN; e_size_idx <= eSIZE_MAX; e_size_idx++)
					{
						if (kaui32_matrix_size[e_size_idx] == ui32_matrix_size)
						{
							e_size = e_size_idx;
						}
					}
				}
				if (argc == 2) /* If only the size is configured, then all experiments should be executed with this size */
				{
					b32_time_exp = TRUE;
					b32_dc_exp = TRUE;
					b32_ut_exp = TRUE;
				}
			}

			if (strcmp(pstr_arg, "-f=all") == 0)
			{
				memcpy(pab32_selected_tech, &ab32_b_tech_all[0], TECH_MAX * sizeof(bool32_t));
			}
			else if (strcmp(pstr_arg, "-f=best") == 0)
			{
				memcpy(pab32_selected_tech, &ab32_b_tech_best[0], TECH_MAX * sizeof(bool32_t));
			}
			else if (strcmp(pstr_arg, "-f=best_dc") == 0)
			{
				memcpy(pab32_selected_tech, &ab32_b_tech_best_dc[0], TECH_MAX * sizeof(bool32_t));
			}
			else if (strcmp(pstr_arg, "-f=best_dc_time") == 0)
			{
				memcpy(pab32_selected_tech, &ab32_b_tech_best_dc_time[0], TECH_MAX * sizeof(bool32_t));
			}
			else
			{
				;
			}

			if (strcmp(argv[ui32_idx], "-h") == 0)
			{
				b32_help = TRUE;
				print_help_commands(argv[0]);
			}
		}
	}
	else
	{
		print_help_commands(argv[0]);
	}

	/* If not experiment explicitly selected, then all experiments are selected by default */
	if ((b32_time_exp == FALSE) && (b32_dc_exp == FALSE) && (b32_ut_exp == FALSE) && (b32_help == FALSE))
	{
		b32_time_exp = TRUE;
		b32_dc_exp = TRUE;
		b32_ut_exp = TRUE;
	}

	*pb32_time_exp = b32_time_exp;
	*pb32_dc_exp = b32_dc_exp;
	*pb32_ut_exp = b32_ut_exp;

	return e_size;
}

static void_t print_help_commands(const char *pstr_exec_name)
{
	printf("\n Help (arguments):");
	printf("\n\t -u  \t Execute unit test experiments");
	printf("\n\t -t  \t Execute time measurement experiments");
	printf("\n\t -dc \t Execute Diagnostic Coverage (DC) measurement experiments");
	printf("\n\t -s=size \t Maximum size of images. Size can be one of those values: 20, 40, 80, 160, 320, 640");
	printf("\n\t -f=selection \t Filter techniques to the given selection: all, best, best_dc, best_dc_time");
	printf("\n\t Note: No argument means all experiments");
	printf("\n\t Example:");
	printf("\n\t\t %s -t -s=320 -f=best_dc", pstr_exec_name);
	printf("\n\t\t\t  Perform time measurements for all techniques that provide best DC (100%%) (the selection) with matrix size up to 320x320");
}

/******************************************************************************
**				Name:    execute_unit_test
******************************************************************************/
/*!
** @brief  Execute unit-test for all technique implementations with respect to
**         the golden implementation (no diagnostic coverage implementation)
**
** @param[in,out] paf32_ma  Pointer to matrix A used in the unit tests
** @param[in,out] paf32_mb  Pointer to matrix B used in the unit tests
** @param[in,out] paf32_mc  Pointer to matrix C used in the unit tests
** @param[in,out] paf32_mc  Pointer to matrix C used as reference in the unit tests
**
** @return  bool32_t Unit test execution results
** @retval  true  OK
** @retval  false FAIL
**
******************************************************************************/
static bool32_t execute_unit_test(bool32_t ab32_selected_tech[TECH_MAX], float32_t* const paf32_ma, float32_t* const paf32_mb, float32_t* const paf32_mc, float32_t* const paf32_mc_ref)
{
	bool32_t b_result = TRUE,
		b32_cmp;
	uint32_t ui32_idx;
	e_enum_technique e_tech;
	int32_t i32_ret;

	/* 1. Check input parameters */
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);
	assert(paf32_mc_ref != NULL);

	/* 2. Initialize matrix values */
	// Matrix A
#ifdef Square_mtrx
	for (ui32_idx = 0u; ui32_idx < (MAX_DIM * MAX_DIM); ui32_idx++)
	{
		paf32_ma[ui32_idx] = (float32_t)(ui32_idx + 1u);
		paf32_mb[ui32_idx] = (float32_t)(ui32_idx + 1u) + 10.0f;
		paf32_mc[ui32_idx] = 0.0f;
		paf32_mc_ref[ui32_idx] = 0.0f;
	}
	smm_no_dc(MAX_DIM, MAX_DIM, MAX_DIM, 1.0f, &paf32_ma[0], &paf32_mb[0], &paf32_mc_ref[0]);
#else
	for (ui32_idx = 0u; ui32_idx < (M * K); ui32_idx++)
	{
		paf32_ma[ui32_idx] = (float32_t)(ui32_idx + 1u);
	}
	// Matrix B
	for (ui32_idx = 0u; ui32_idx < (K * N); ui32_idx++)
	{
		paf32_mb[ui32_idx] = (float32_t)(ui32_idx + 1u) + 10.0f;
	}
	// Matrix C
	for (ui32_idx = 0u; ui32_idx < (M * N); ui32_idx++)
	{
		paf32_mc[ui32_idx] = 0.0f;
		paf32_mc_ref[ui32_idx] = 0.0f;
	}

	smm_no_dc(M, N, K, 1.0f, &paf32_ma[0], &paf32_mb[0], &paf32_mc_ref[0]);
#endif




	/* 3. Execute unit tests */   // TECH_MAX TECH_INTEL_COMB
	for (e_tech = TECH_INTEL_XOR_EXTERNAL; e_tech < TECH_INTEL_COMB; e_tech++)
	{
		if (ab32_selected_tech[e_tech])
		{
#ifdef Square_mtrx
			matrix2zeros(&paf32_mc[0], MAX_DIM, MAX_DIM);
			ptr_fn_smm_technique[e_tech](MAX_DIM, MAX_DIM, MAX_DIM, 1.0f, &paf32_ma[0], &paf32_mb[0], &paf32_mc[0]);
			i32_ret = memcmp((const void_t*)paf32_mc_ref, (const void_t*)paf32_mc, (size_t)(MAX_DIM * MAX_DIM * sizeof(float32_t)));
#else
			matrix2zeros(&paf32_mc[0], M, N);
			ptr_fn_smm_technique[e_tech](M, N, K, 1.0f, &paf32_ma[0], &paf32_mb[0], &paf32_mc[0]);
			i32_ret = memcmp((const void_t*)paf32_mc_ref, (const void_t*)paf32_mc, (size_t)(M * N * sizeof(float32_t)));
#endif
			b32_cmp = (i32_ret == 0);
			printf("\n\t Unit Test %2u / %2u (%25s): %4s", (e_tech + 1u), (TECH_MAX + 1u), pstr_technique[e_tech], b32_cmp ? "OK" : "FAIL");
		}
	}

	return b_result;
}

static int32_t measure_time(e_enum_size_2d e_size_max, bool32_t ab32_selected_tech[TECH_MAX], float32_t* const paf32_ma, float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	e_enum_technique e_tech;
	uint32_t ui32_idx,
		ui32_t_loop;
	float32_t f32_alpha = 1.0f,
		f32_time_min,
		f32_time_max,
		f32_time_avg;
	float32_t af32_time_ref[eSIZE_MAX];
	e_enum_size_2d e_size;
	DEF_TIME_VAR(tmr_start);
	DEF_TIME_VAR(tmr_end);
	DEF_TIME_VAR(tmr_start_exp);
	DEF_TIME_VAR(tmr_end_exp);
	float64_t time_interval;


#if defined __linux__ || defined _WIN32
	FILE *p_file;
	char str_file_name[100u];
	errno_t err;
	time_t time_now = time(NULL);
#endif

	/* 1. Initialize matrixes */
#ifdef Square_mtrx
	matrix2rand(&paf32_ma[0], MAX_DIM, MAX_DIM);
	matrix2rand(&paf32_mb[0], MAX_DIM, MAX_DIM);
	matrix2zeros(&paf32_mc[0], MAX_DIM, MAX_DIM);
#else
	matrix2rand(&paf32_ma[0], M, K);
	matrix2rand(&paf32_mb[0], K, N);
	matrix2zeros(&paf32_mc[0], M, N);
#endif

	/* 2. Open output file (csv) */
#ifdef _WIN32
	struct tm time_info;
	localtime_s(&time_info, &time_now);
	strftime(str_file_name, sizeof(str_file_name), "time_csv_%y_%m_%d__%A__%H_%M_%S.csv", &time_info);
	if ((err = fopen_s(&p_file, str_file_name, "w+")) != 0)
	{
		fprintf(stderr, "cannot open file '%s': %u\n", str_file_name, err);
		return EXIT_FAILURE;
	}

	if (!p_file)
	{
		perror("File opening failed");
		return EXIT_FAILURE;
	}
	fprintf(p_file, "Technique, Size, Time");
#endif


#ifdef __linux__
	struct tm *time_info;
	time_info = localtime(&time_now);
	strftime(str_file_name, sizeof(str_file_name), "time_csv_%y_%m_%d_%A_%H_%M_%S.csv", time_info);
	if ((p_file = fopen(str_file_name, "w+")) == NULL)
	{
		fprintf(stderr, "cannot open file '%s'\n", str_file_name);
		return EXIT_FAILURE;
	}

	if (!p_file)
	{
		perror("File opening failed");
		return EXIT_FAILURE;
	}
	//fprintf(p_file, "Technique, Size, Time");
#endif


	/* 3. Execute time measurement */
	e_size_max = (e_size_max < eSIZE_MAX) ? e_size_max : (eSIZE_MAX - 1u);
	GET_TIME(tmr_start_exp);  //TECH_INTEL_COMB
	for (e_tech = TECH_INTEL_NO_DC; e_tech < TECH_INTEL_COMB; e_tech++)
	{
		if (ab32_selected_tech[e_tech] || (e_tech == TECH_NONE))
		{
			printf("\n\t Experiment %2u / %2u (%25s):", (e_tech + 1u), (TECH_MAX + 1u), pstr_technique[e_tech]);
			for (e_size = eSIZE_40_40; e_size <= e_size_max; e_size++)
			{
#ifdef Square_mtrx
				printf("\n\t\t [%3u x %3u],", kaui32_matrix_size[e_size], kaui32_matrix_size[e_size]);
#else
				printf("\n\t\t A=[%3u x %3u], B=[%3u x %3u],", M, K, K, N);
#endif
				f32_time_min = FLT_MAX;
				f32_time_max = 0.0f;
				f32_time_avg = 0.0f;
#if defined __linux__ || defined _WIN32
				fprintf(p_file, "\n%s,%u,", pstr_technique[e_tech], kaui32_matrix_size[e_size]);
#endif
				for (ui32_idx = 0u; ui32_idx < MEASUREMENT_LOOPS; ui32_idx++)
				{
					GET_TIME(tmr_start);
					for (ui32_t_loop = 0u; ui32_t_loop < TIME_MEASUREMENT_LOOPS; ui32_t_loop++)
					{
#ifdef Square_mtrx
						ptr_fn_smm_technique[e_tech](kaui32_matrix_size[e_size], kaui32_matrix_size[e_size], kaui32_matrix_size[e_size], f32_alpha, (float32_t* const)paf32_ma, (float32_t* const)paf32_mb, (float32_t* const)paf32_mc);
#else
						ptr_fn_smm_technique[e_tech](M, N, K, f32_alpha, (float32_t* const)paf32_ma, (float32_t* const)paf32_mb, (float32_t* const)paf32_mc);
#endif
					}
					GET_TIME(tmr_end);
					GET_TIME_DIFF(tmr_start, tmr_end, time_interval);
					time_interval *= ((float64_t)TIME_SEC2USEC) / (float64_t)TIME_MEASUREMENT_LOOPS;
					printf("%15.4f,", time_interval);
#if defined __linux__ || defined _WIN32
					fprintf(p_file, "%15.4f,", time_interval);
#endif

#ifdef Square_mtrx
					matrix2zeros(&paf32_mc[0], MAX_DIM, MAX_DIM);
#else
					matrix2zeros(&paf32_mc[0], M, N);
#endif
					// The min, max and average values will be evaluated in excel instead of in the Zynq.
					//f32_time_min = ((float32_t)time_interval < f32_time_min) ? (float32_t)time_interval : f32_time_min;
					//f32_time_max = ((float32_t)time_interval > f32_time_max) ? (float32_t)time_interval : f32_time_max;
					//f32_time_avg += (float32_t)time_interval;
				}
				f32_time_avg /= (float_t)MEASUREMENT_LOOPS;
				if (e_tech == TECH_NONE)
				{
					af32_time_ref[e_size] = f32_time_avg;
				}
				//printf("\n\t\t [%3u x %3u] min = %15.1f [usec], max = %15.1f [usec], avg = %15.1f [usec], ratio_avg = x%5.2f", kaui32_matrix_size[e_size], kaui32_matrix_size[e_size], f32_time_min, f32_time_max, f32_time_avg, (af32_time_ref[e_size] > 0.0f) ? (f32_time_avg / af32_time_ref[e_size]) : 0.0f);
				//printf("=PROMEDIO(B%u:K%u),",e_tech+e_size+25+e_tech*e_size,e_tech+e_size+25+e_tech*e_size);
				//printf("=DESVEST.M(B%u:K%u),",e_tech+e_size+25+e_tech*e_size,e_tech+e_size+25+e_tech*e_size);
				//printf("=MAX(B%u:K%u),",e_tech+e_size+25+e_tech*e_size,e_tech+e_size+25+e_tech*e_size);
				//printf("=MIN(B%u:K%u),",e_tech+e_size+25+e_tech*e_size,e_tech+e_size+25+e_tech*e_size);
				//printf("%u,",(af32_time_ref[e_size] > 0.0f) ? (f32_time_avg / af32_time_ref[e_size]) : 0.0f);
			}
		}
		else
		{
			//printf(" -> Skipped, not selected in techniques filter");
		}
	}
#if defined __linux__ || defined _WIN32
	fclose(p_file);
#endif
	GET_TIME(tmr_end_exp);
	GET_TIME_DIFF(tmr_start_exp, tmr_end_exp, time_interval);
	printf("\n\t\t Experiments executed in %10.f [sec]", time_interval);

	return EXIT_SUCCESS;
}

static int32_t measure_dc__error_bit_parallelized(e_enum_size_2d e_size_max, bool32_t ab32_selected_tech[TECH_MAX], bool32_t b32_double_error,
	float32_t* const paf32_ma, float32_t* const paf32_mb, float32_t* const paf32_mc,
	float32_t* const paf32_ma_fi, float32_t* const paf32_mb_fi,
	char *argv[])
{

#if defined Square_mtrx
	uint32_t ui32_combinations;
	e_enum_size_2d e_size;
#else
	uint32_t ui32_combinations_a, ui32_combinations_b, ui32_idx_bit_aux, launch_number;
	ui32_combinations_a = (uint32_t)strtoul(argv[5], NULL, 10);
	ui32_combinations_b = (uint32_t)strtoul(argv[6], NULL, 10);
	ui32_idx_bit_aux = (uint32_t)strtoul(argv[7], NULL, 10);
	launch_number = (uint32_t)strtoul(argv[8], NULL, 10);
	printf("\n\t A=%u\t B=%u\t BIT=%u\t launch_number=%u", ui32_combinations_a, ui32_combinations_b, ui32_idx_bit_aux, launch_number);
#endif

	e_enum_technique e_tech;
	//e_enum_size_2d e_size;
	size_t size_a, size_b, size_c, size;
	uint32_t ui32_idx_bit,
		//ui32_matrix_size,
		ui32_dc_cnt_all = 0u,
		ui32_dc_cnt = 0u,
		aui32_dc_value[e_FI_VAR_MAX];
	float32_t f32_alpha = 1.0f;
	//f32_dc;
//DEF_TIME_VAR(tmr_start);
//DEF_TIME_VAR(tmr_end);
	DEF_TIME_VAR(tmr_start_exp);
	DEF_TIME_VAR(tmr_end_exp);
	float64_t time_interval;


#ifndef Square_mtrx
	size_a = M * K * sizeof(float32_t);
	size_b = N * K * sizeof(float32_t);
	size_c = M * N * sizeof(float32_t);
#endif

	/* 1. Execute DC measurement */
	e_size_max = (e_size_max < eSIZE_MAX) ? e_size_max : (eSIZE_MAX - 1u);
	GET_TIME(tmr_start_exp);

#ifdef Square_mtrx
	matrix2zeros(&paf32_ma_fi[0u], MAX_DIM, MAX_DIM);
	matrix2zeros(&paf32_mb_fi[0u], MAX_DIM, MAX_DIM);
	matrix2zeros(&paf32_ma[0u], MAX_DIM, MAX_DIM);
	matrix2zeros(&paf32_mb[0u], MAX_DIM, MAX_DIM);
	matrix2zeros(&paf32_mc[0u], MAX_DIM, MAX_DIM);
#else
	matrix2zeros(&paf32_ma_fi[0u], M, K);
	matrix2zeros(&paf32_mb_fi[0u], K, N);
	matrix2zeros(&paf32_ma[0u], M, K);
	matrix2zeros(&paf32_mb[0u], K, N);
	matrix2zeros(&paf32_mc[0u], M, N);
#endif

	/* Fault injection campaing */
#ifndef Square_mtrx
	uint32_t ui32_comb_a_max = ui32_idx_bit_aux + ui32_combinations_a;
	uint32_t ui32_comb_b_max = ui32_idx_bit_aux + ui32_combinations_b;
	memcpy(&paf32_ma_fi[0u], &paf32_ma[0u], size_a);
#endif

	//TECH_INTEL_XOR_CRC TECH_INTEL_ONES_INTERNAL TECH_INTEL_XOR_EXTERNAL TECH_INTEL_COMB
	for (e_tech = TECH_INTEL_XOR_EXTERNAL; e_tech < TECH_INTEL_COMB; e_tech++)
	{
		if (ab32_selected_tech[e_tech])
		{
#ifdef _WIN32
			FILE *p_file;
			FILE *p_file_idx_fi;
			char str_file_name[200u];
			char str_file_name_aux[100u];
			char str_file_name_idx_fi[200u];

			errno_t err;
			time_t time_now = time(NULL);
			struct tm time_info;

			/* 1. Open output file (csv) */

			localtime_s(&time_info, &time_now);
			if (b32_double_error)
			{
				strftime(str_file_name_aux, sizeof(str_file_name_aux), "dc_double_bit_csv_%y_%m_%d__%A__%H_%M_%S.csv", &time_info);
			}
			else
			{
				strftime(str_file_name_aux, sizeof(str_file_name_aux), "dc_single_bit_csv_%d_%H_%M_%S", &time_info);
			}

#ifdef Square_mtrx
			snprintf(str_file_name, 200u, "%s_fi.csv", pstr_technique[e_tech]);
			snprintf(str_file_name_idx_fi, 200u, "%s_fi_idx.csv", pstr_technique[e_tech]);
#else
			snprintf(str_file_name, 200u, "%s_fi_%u.csv", pstr_technique[e_tech], launch_number);
			snprintf(str_file_name_idx_fi, 200u, "%s_fi_%u_idx.csv", pstr_technique[e_tech], launch_number);
#endif


			if ((err = fopen_s(&p_file, str_file_name, "w+")) != 0)
			{
				fprintf(stderr, "cannot open file '%s': %u\n", str_file_name, err);
				return EXIT_FAILURE;
			}

			if ((err = fopen_s(&p_file_idx_fi, str_file_name_idx_fi, "w+")) != 0)
			{
				fprintf(stderr, "cannot open file '%s': %u\n", str_file_name_idx_fi, err);
				return EXIT_FAILURE;
			}
#endif

#ifdef __linux__
			FILE *p_file;
			FILE *p_file_idx_fi;
			char str_file_name[200u];
			char str_file_name_aux[100u];
			char str_file_name_idx_fi[200u];

			errno_t err;
			time_t time_now = time(NULL);
			struct tm *time_info;

			/* 1. Open output file (csv) */
			time_now = time(NULL);
			time_info = localtime(&time_now);

			if (b32_double_error)
			{
				strftime(str_file_name_aux, sizeof(str_file_name), "dc_double_bit_csv_%y_%m_%d__%A__%H_%M_%S.csv", time_info);
			}
			else
			{
				strftime(str_file_name_aux, sizeof(str_file_name), "%m_%d_%H_%M_%S", time_info);
			}


#ifdef Square_mtrx
			snprintf(str_file_name, 200u, "%s_fi.csv", pstr_technique[e_tech]);
			snprintf(str_file_name_idx_fi, 200u, "%s_fi_idx.csv", pstr_technique[e_tech]);
#else
			snprintf(str_file_name, 200u, "%s_fi_%u.csv", pstr_technique[e_tech], launch_number);
			snprintf(str_file_name_idx_fi, 200u, "%s_fi_%u_idx.csv", pstr_technique[e_tech], launch_number);
#endif


			if ((p_file = fopen(str_file_name, "w+")) == NULL)
			{
				fprintf(stderr, "cannot open file '%s'\n", str_file_name);
				return EXIT_FAILURE;
			}

			if ((p_file_idx_fi = fopen(str_file_name_idx_fi, "w+")) == NULL)
			{
				fprintf(stderr, "cannot open file '%s'\n", str_file_name_idx_fi);
				return EXIT_FAILURE;
			}
#endif


#ifdef Square_mtrx
			e_size_max = (e_size_max < eSIZE_MAX) ? e_size_max : (eSIZE_MAX - 1u);

			for (e_size = eSIZE_MIN; e_size <= e_size_max; e_size++)
			{ //eSIZE_MIN eSIZE_20_20 eSize_40_40 e_size_max
				// Restart the value of detected error for each matrix size experiment
				ui32_dc_cnt = 0u;
				ui32_dc_cnt_all = 0u;
				size = kaui32_matrix_size[e_size] * kaui32_matrix_size[e_size] * sizeof(float32_t) * 8;
				ui32_combinations = (kaui32_matrix_size[e_size] * kaui32_matrix_size[e_size]) * sizeof(uint32_t) * 8;

				printf("\n\t\t [%3u x %3u],", kaui32_matrix_size[e_size], kaui32_matrix_size[e_size]);

				/* 1. Store the Execution Signature (ES) for comparing with the value obtained after the fault injection*/
				aui32_dc_value[e_FI_VAR_NONE] = ptr_fn_smm_technique[e_tech](kaui32_matrix_size[e_size], kaui32_matrix_size[e_size], kaui32_matrix_size[e_size], f32_alpha, (float32_t* const)paf32_ma, (float32_t* const)paf32_mb, (float32_t* const)paf32_mc);
				for (ui32_idx_bit = 0u; ui32_idx_bit < ui32_combinations; ui32_idx_bit++) {
					mem_fi(&paf32_ma_fi[0], ui32_idx_bit);
					mem_fi(&paf32_mb_fi[0], ui32_idx_bit);

					aui32_dc_value[e_FI_VAR_A] = ptr_fn_smm_technique[e_tech](kaui32_matrix_size[e_size], kaui32_matrix_size[e_size], kaui32_matrix_size[e_size], f32_alpha, (float32_t* const)paf32_ma_fi, (float32_t* const)paf32_mb, (float32_t* const)paf32_mc);
					aui32_dc_value[e_FI_VAR_B] = ptr_fn_smm_technique[e_tech](kaui32_matrix_size[e_size], kaui32_matrix_size[e_size], kaui32_matrix_size[e_size], f32_alpha, (float32_t* const)paf32_ma, (float32_t* const)paf32_mb_fi, (float32_t* const)paf32_mc);

					ui32_dc_cnt_all += 2u;
					//ui32_dc_cnt = (aui32_dc_value[e_FI_VAR_NONE] != aui32_dc_value[e_FI_VAR_A]) ? (ui32_dc_cnt + 1u) : ui32_dc_cnt;
					//ui32_dc_cnt = (aui32_dc_value[e_FI_VAR_NONE] != aui32_dc_value[e_FI_VAR_B]) ? (ui32_dc_cnt + 1u) : ui32_dc_cnt;

					if (aui32_dc_value[e_FI_VAR_NONE] != aui32_dc_value[e_FI_VAR_A]) {
						ui32_dc_cnt += 1u;
					}
					else {
						fprintf(p_file_idx_fi, "%u,", ui32_idx_bit);
					}
					if (aui32_dc_value[e_FI_VAR_NONE] != aui32_dc_value[e_FI_VAR_B]) {
						ui32_dc_cnt += 1u;
					}
					else {
						fprintf(p_file_idx_fi, "%u,", ui32_idx_bit);
					}

					mem_fi(&paf32_ma_fi[0], ui32_idx_bit);
					mem_fi(&paf32_mb_fi[0], ui32_idx_bit);
				}
				fprintf(p_file, "\n%s, %u, %u, %u", pstr_technique[e_tech], kaui32_matrix_size[e_size], ui32_dc_cnt_all, ui32_dc_cnt);
			}
#else
			// 1. Store the Execution Signature (ES) for comparing with the value obtained after the fault injection
			aui32_dc_value[e_FI_VAR_NONE] = ptr_fn_smm_technique[e_tech](M, N, K, f32_alpha, (float32_t* const)paf32_ma, (float32_t* const)paf32_mb, (float32_t* const)paf32_mc);
			fprintf(p_file, "diagnostic_technique,detected_errors_a,total_detected_errors,number_of_fi,M, N, K,idx_fi_initial,idx_fi_final_a,idx_fi_final_b,launch number,");
			fprintf(p_file, "\n%s, ", pstr_technique[e_tech]);
			for (ui32_idx_bit = ui32_idx_bit_aux; ui32_idx_bit < ui32_comb_a_max; ui32_idx_bit++) {
				mem_fi(&paf32_ma_fi[0], ui32_idx_bit);
				aui32_dc_value[e_FI_VAR_A] = ptr_fn_smm_technique[e_tech](M, N, K, f32_alpha, (float32_t* const)paf32_ma_fi, (float32_t* const)paf32_mb, (float32_t* const)paf32_mc);
				if (aui32_dc_value[e_FI_VAR_NONE] != aui32_dc_value[e_FI_VAR_A]) {
					ui32_dc_cnt += 1u;
				}
				else {
					fprintf(p_file_idx_fi, "%u,", ui32_idx_bit);

				}
				mem_fi(&paf32_ma_fi[0], ui32_idx_bit);
			}

			fprintf(p_file, "%u,", ui32_dc_cnt);
			fprintf(p_file_idx_fi, "\n");
			for (ui32_idx_bit = ui32_idx_bit_aux; ui32_idx_bit < ui32_comb_b_max; ui32_idx_bit++)
			{
				mem_fi(&paf32_mb_fi[0], ui32_idx_bit);
				aui32_dc_value[e_FI_VAR_B] = ptr_fn_smm_technique[e_tech](M, N, K, f32_alpha, (float32_t* const)paf32_ma, (float32_t* const)paf32_mb_fi, (float32_t* const)paf32_mc);
				if (aui32_dc_value[e_FI_VAR_NONE] != aui32_dc_value[e_FI_VAR_B]) {
					ui32_dc_cnt += 1u;
				}
				else {
					fprintf(p_file_idx_fi, "%u,", ui32_idx_bit);
				}
				mem_fi(&paf32_mb_fi[0], ui32_idx_bit);
			}
			fprintf(p_file, "%u,%u,%d,%d,%d,%u,%u,%u,%u", ui32_dc_cnt, (ui32_combinations_a + ui32_combinations_b), M, N, K, ui32_idx_bit_aux, ui32_comb_a_max, ui32_comb_b_max, launch_number);

#endif
			// Close the file where the result are been stored
			fclose(p_file);
			fclose(p_file_idx_fi);
		}
	}


	GET_TIME(tmr_end_exp);
	GET_TIME_DIFF(tmr_start_exp, tmr_end_exp, time_interval);
	printf("\n Experiments executed in %10.8lf [sec]", time_interval);
	return EXIT_SUCCESS;
}

static int32_t measure_dc__error_random_value(e_enum_size_2d e_size_max, bool32_t ab32_selected_tech[TECH_MAX],
	float32_t* const paf32_ma, float32_t* const paf32_mb, float32_t* const paf32_mc,
	float32_t* const paf32_ma_fi, float32_t* const paf32_mb_fi,
	float32_t* const paf32_ma_rand, float32_t* const paf32_mb_rand, float32_t* const paf32_mc_rand)
{
	e_enum_technique e_tech;
	e_enum_size_2d e_size;
	size_t size_a;
	size_t size_b;

	uint32_t ui32_idx_value,
		ui32_matrix_size,
		ui32_dc_cnt_all = 0u,
		ui32_dc_cnt = 0u,
		ui32_combinations_a = 0u,
		ui32_combinations_b = 0u,
		aui32_dc_value[e_FI_VAR_MAX];
	uint32_t *pui32_a,
		*pui32_b;
	float32_t f32_alpha = 1.0f,
		f32_dc;
	DEF_TIME_VAR(tmr_start);
	DEF_TIME_VAR(tmr_end);
	DEF_TIME_VAR(tmr_start_exp);
	DEF_TIME_VAR(tmr_end_exp);
	float64_t time_interval;
#ifdef _WIN32
	FILE *p_file;
	char str_file_name[200u];
	errno_t err;
	time_t time_now = time(NULL);
	struct tm time_info;

	/* 1. Open output file (csv) */
	localtime_s(&time_info, &time_now);
	strftime(str_file_name, sizeof(str_file_name), "dc_random_value_csv_%y_%m_%d__%A__%H_%M_%S.csv", &time_info);
	if ((err = fopen_s(&p_file, str_file_name, "w+")) != 0)
	{
		fprintf(stderr, "cannot open file '%s': %u\n", str_file_name, err);
		return EXIT_FAILURE;
	}
	fprintf(p_file, "Technique, Size, DC");
#endif

	/* 1. Initialize matrixes */
	matrix2rand(&paf32_ma[0], MAX_DIM, MAX_DIM);
	matrix2rand(&paf32_mb[0], MAX_DIM, MAX_DIM);
	matrix2zeros(&paf32_mc[0], MAX_DIM, MAX_DIM);

	matrix2rand(&paf32_ma_rand[0], MAX_DIM, MAX_DIM);
	matrix2rand(&paf32_mb_rand[0], MAX_DIM, MAX_DIM);
	matrix2zeros(&paf32_mc_rand[0], MAX_DIM, MAX_DIM);

	/* 3. Execute DC measurement */
	e_size_max = (e_size_max < eSIZE_MAX) ? e_size_max : (eSIZE_MAX - 1u);
	GET_TIME(tmr_start_exp);
	for (e_tech = TECH_NONE; e_tech < TECH_MAX; e_tech++)
	{
		if (ab32_selected_tech[e_tech])
		{
			printf("\n\t Experiment %2u / %2u (%25s): ", (e_tech + 1u), (TECH_MAX + 1u), pstr_technique[e_tech]);
			for (e_size = eSIZE_MIN; e_size <= e_size_max; e_size++)
			{
				size_a = M * K * sizeof(float32_t);
				size_b = K * N * sizeof(float32_t);
				ui32_matrix_size = kaui32_matrix_size[e_size];
				ui32_dc_cnt = 0u;
				ui32_dc_cnt_all = 0u;

				matrix2zeros(&paf32_ma[0u], M, K);
				matrix2zeros(&paf32_mb[0u], K, N);
				matrix2zeros(&paf32_mc[0u], M, N);

				ui32_combinations_a = (M*K);
				//Before: printf("\n\t\t [%3u x %3u]  Execution progress of 2 x (%6u) = %8u random value replacement error = [", ui32_matrix_size, ui32_matrix_size, ui32_combinations, (2u * ui32_combinations));
				printf("\n\t\t  A =[%3u x %3u] B =[%3u x %3u],  Execution progress of 2 x (%6u x %6u) = %12u single-bit error injections = [", M, K, K, N, ui32_combinations_a, ui32_combinations_a, (2u * ui32_combinations_a * ui32_combinations_a));
				GET_TIME(tmr_start);
				aui32_dc_value[e_FI_VAR_NONE] = ptr_fn_smm_technique[e_tech](ui32_matrix_size, ui32_matrix_size, ui32_matrix_size, f32_alpha, (float32_t* const)paf32_ma, (float32_t* const)paf32_mb, (float32_t* const)paf32_mc);
				for (ui32_idx_value = 0u; ui32_idx_value < ui32_combinations_a; ui32_idx_value++)
				{
					memcpy(&paf32_ma_fi[0u], &paf32_ma[0u], size_a);
					pui32_a = (uint32_t *)&paf32_ma_fi[ui32_idx_value];
					*pui32_a = (uint32_t)rand();

					aui32_dc_value[e_FI_VAR_A] = ptr_fn_smm_technique[e_tech](M, K, N, f32_alpha, (float32_t* const)paf32_ma_fi, (float32_t* const)paf32_mb, (float32_t* const)paf32_mc);

					ui32_dc_cnt_all += 2u;
					ui32_dc_cnt = (aui32_dc_value[e_FI_VAR_NONE] != aui32_dc_value[e_FI_VAR_A]) ? (ui32_dc_cnt + 1u) : ui32_dc_cnt;
					/*if ((ui32_idx_value % (ui32_combinations / 10u) == 0u))
					{
					printf("%1u", ui32_idx_value / (ui32_combinations / 10u));
					}*/
				}

				for (ui32_idx_value = 0u; ui32_idx_value < ui32_combinations_b; ui32_idx_value++)
				{
					memcpy(&paf32_mb_fi[0u], &paf32_mb[0u], size_b);
					pui32_b = (uint32_t *)&paf32_mb_fi[ui32_idx_value];
					*pui32_b = (uint32_t)rand();

					aui32_dc_value[e_FI_VAR_B] = ptr_fn_smm_technique[e_tech](M, K, N, f32_alpha, (float32_t* const)paf32_ma, (float32_t* const)paf32_mb_fi, (float32_t* const)paf32_mc);

					ui32_dc_cnt_all += 2u;
					ui32_dc_cnt = (aui32_dc_value[e_FI_VAR_NONE] != aui32_dc_value[e_FI_VAR_B]) ? (ui32_dc_cnt + 1u) : ui32_dc_cnt;
					/*if ((ui32_idx_value % (ui32_combinations / 10u) == 0u))
					{
					printf("%1u", ui32_idx_value / (ui32_combinations / 10u));
					}*/
				}
				f32_dc = (100.0f * (float32_t)ui32_dc_cnt) / ((float32_t)ui32_dc_cnt_all);
				GET_TIME(tmr_end);
				GET_TIME_DIFF(tmr_start, tmr_end, time_interval);
				// Before: printf("]; Diagnostic coverage = %5.2f %%; Exec Time = %6.2f [sec]", f32_dc, time_interval);
				printf("%5.2f, Exec Time = %6.2f", f32_dc, time_interval);
#ifdef _WIN32
				fprintf(p_file, "\n%s,%u,%f", pstr_technique[e_tech], kaui32_matrix_size[e_size], f32_dc);
#endif
			}
		}
		else
		{
			//printf(" -> Skipped, not selected in techniques filter");
		}
	}
#ifdef _WIN32
	fclose(p_file);
#endif

	GET_TIME(tmr_end_exp);
	GET_TIME_DIFF(tmr_start_exp, tmr_end_exp, time_interval);
	printf("\n\t\t Experiments executed in %10.f [sec]", time_interval);

	return EXIT_SUCCESS;
}

static int32_t measure_dc__error_random_values(e_enum_size_2d e_size_max, bool32_t ab32_selected_tech[TECH_MAX], bool32_t b32_consecutive, uint32_t ui32_n_length, uint32_t ui32_n_iterations,
	float32_t* const paf32_ma, float32_t* const paf32_mb, float32_t* const paf32_mc,
	float32_t* const paf32_ma_fi, float32_t* const paf32_mb_fi,
	float32_t* const paf32_ma_rand, float32_t* const paf32_mb_rand, float32_t* const paf32_mc_rand)
{
	e_enum_technique e_tech;
	e_enum_size_2d e_size;
	size_t size;
	uint32_t ui32_idx_iteration,
		ui32_idx_length,
		ui32_matrix_size,
		ui32_dc_cnt_all = 0u,
		ui32_dc_cnt = 0u,
		aui32_dc_value[e_FI_VAR_MAX];
	float32_t f32_alpha = 1.0f,
		f32_dc;
	DEF_TIME_VAR(tmr_start);
	DEF_TIME_VAR(tmr_end);
	DEF_TIME_VAR(tmr_start_exp);
	DEF_TIME_VAR(tmr_end_exp);
	float64_t time_interval;
#ifdef _WIN32
	FILE *p_file;
	char str_file_name[100u];
	errno_t err;
	time_t time_now = time(NULL);
	struct tm time_info;

	/* 1. Open output file (csv) */
	localtime_s(&time_info, &time_now);
	strftime(str_file_name, sizeof(str_file_name), "dc_random_error_csv_%y_%m_%d__%A__%H_%M_%S.csv", &time_info);
	if ((err = fopen_s(&p_file, str_file_name, "w+")) != 0)
	{
		fprintf(stderr, "cannot open file '%s': %u\n", str_file_name, err);
		return EXIT_FAILURE;
	}
	fprintf(p_file, "Technique, Size, DC");
#endif

	/* 1. Initialize matrixes */
	matrix2rand(&paf32_ma[0], MAX_DIM, MAX_DIM);
	matrix2rand(&paf32_mb[0], MAX_DIM, MAX_DIM);
	matrix2zeros(&paf32_mc[0], MAX_DIM, MAX_DIM);

	matrix2rand(&paf32_ma_rand[0], MAX_DIM, MAX_DIM);
	matrix2rand(&paf32_mb_rand[0], MAX_DIM, MAX_DIM);
	matrix2zeros(&paf32_mc_rand[0], MAX_DIM, MAX_DIM);

	/* 3. Execute DC measurement */
	e_size_max = (e_size_max < eSIZE_MAX) ? e_size_max : (eSIZE_MAX - 1u);
	GET_TIME(tmr_start_exp);
	for (e_tech = TECH_NONE; e_tech < TECH_MAX; e_tech++)
	{
		if (ab32_selected_tech[e_tech])
		{
			printf("\n\t Experiment %2u / %2u (%25s): ", (e_tech + 1u), (TECH_MAX + 1u), pstr_technique[e_tech]);
			GET_TIME(tmr_start);
			for (e_size = eSIZE_MIN; e_size <= e_size_max; e_size++)
			{
				printf("\n [%3u x %3u]  Execution progress of %6u random error replacements of %2u variables at %s positions. Execution percentage = [", kaui32_matrix_size[e_size], kaui32_matrix_size[e_size], ui32_n_iterations, ui32_n_length, (b32_consecutive) ? "consecutive" : "random");
				for (ui32_idx_length = 0u; ui32_idx_length < ui32_n_length; ui32_idx_length++)
				{
					for (ui32_idx_iteration = 0u; ui32_idx_iteration < ui32_n_iterations; ui32_idx_iteration++)
					{
						size = kaui32_matrix_size[e_size] * kaui32_matrix_size[e_size] * sizeof(float32_t);
						ui32_matrix_size = kaui32_matrix_size[e_size];
						ui32_dc_cnt = 0u;
						ui32_dc_cnt_all = 0u;

						if (ui32_idx_iteration == 0u)
						{
							matrix2zeros(&paf32_ma[0u], MAX_DIM, MAX_DIM);
							matrix2zeros(&paf32_mb[0u], MAX_DIM, MAX_DIM);
							matrix2zeros(&paf32_mc[0u], MAX_DIM, MAX_DIM);
						}
						else
						{
							memcpy(&paf32_ma[0u], &paf32_ma_rand[0], size);
							memcpy(&paf32_mb[0u], &paf32_mb_rand[0], size);
							memcpy(&paf32_mc[0u], &paf32_mc_rand[0], size);
						}

						aui32_dc_value[e_FI_VAR_NONE] = ptr_fn_smm_technique[e_tech](ui32_matrix_size, ui32_matrix_size, ui32_matrix_size, f32_alpha, (float32_t* const)paf32_ma, (float32_t* const)paf32_mb, (float32_t* const)paf32_mc);
						memcpy(&paf32_ma_fi[0u], &paf32_ma[0u], size);
						memcpy(&paf32_mb_fi[0u], &paf32_mb[0u], size);
						mem_fi_random_value(&paf32_ma_fi[0], ui32_matrix_size * ui32_matrix_size, ui32_idx_length, b32_consecutive);
						mem_fi_random_value(&paf32_mb_fi[0], ui32_matrix_size * ui32_matrix_size, ui32_idx_length, b32_consecutive);

						aui32_dc_value[e_FI_VAR_A] = ptr_fn_smm_technique[e_tech](ui32_matrix_size, ui32_matrix_size, ui32_matrix_size, f32_alpha, (float32_t* const)paf32_ma_fi, (float32_t* const)paf32_mb, (float32_t* const)paf32_mc);
						aui32_dc_value[e_FI_VAR_B] = ptr_fn_smm_technique[e_tech](ui32_matrix_size, ui32_matrix_size, ui32_matrix_size, f32_alpha, (float32_t* const)paf32_ma, (float32_t* const)paf32_mb_fi, (float32_t* const)paf32_mc);

						ui32_dc_cnt_all += 2u;
						ui32_dc_cnt = (aui32_dc_value[e_FI_VAR_NONE] != aui32_dc_value[e_FI_VAR_A]) ? (ui32_dc_cnt + 1u) : ui32_dc_cnt;
						ui32_dc_cnt = (aui32_dc_value[e_FI_VAR_NONE] != aui32_dc_value[e_FI_VAR_B]) ? (ui32_dc_cnt + 1u) : ui32_dc_cnt;
					}
					if (((ui32_idx_length) % ((ui32_n_length)) == 0u))
					{
						printf("%1u", ui32_idx_iteration / (ui32_n_iterations / 10u));
					}
				}
				f32_dc = (100.0f * (float32_t)ui32_dc_cnt) / ((float32_t)ui32_dc_cnt_all);
				GET_TIME(tmr_end);
				GET_TIME_DIFF(tmr_start, tmr_end, time_interval);
#ifdef _WIN32
				fprintf(p_file, "\n%s,%u,%f", pstr_technique[e_tech], kaui32_matrix_size[e_size], f32_dc);
#endif
				printf("]; Diagnostic coverage = %5.2f %%; Exec Time = %6.1f [sec]", f32_dc, time_interval);
			}
		}
		else
		{
			//printf(" -> Skipped, not selected in techniques filter");
		}
	}
#ifdef _WIN32
	fclose(p_file);
#endif
	GET_TIME(tmr_end_exp);
	GET_TIME_DIFF(tmr_start_exp, tmr_end_exp, time_interval);
	printf("\n\t\t Experiments executed in %10.f [sec]", time_interval);

	return EXIT_SUCCESS;
}


static void_t mem_fi(float32_t* const paf32_m, uint32_t ui32_bit_idx)
{
	uint32_t ui32_idx_flt = ui32_bit_idx / (sizeof(float32_t)*CHAR_BIT),
		ui32_idx_flt_bit = ui32_bit_idx % (sizeof(float32_t)*CHAR_BIT);
	uint32_t ui32_f_d = *((uint32_t *)&paf32_m[ui32_idx_flt]);

	ui32_f_d ^= (1u << ui32_idx_flt_bit);
	paf32_m[ui32_idx_flt] = *((float32_t *)&ui32_f_d);
}

static void_t mem_fi_random_value(float32_t* const paf32_m, uint32_t ui32_max_dim, uint32_t ui32_n_length, bool32_t b32_consecutive)
{
	uint32_t ui32_idx_error,
		ui32_idx;
	uint32_t *pui32_m = (uint32_t *)paf32_m;

	if (b32_consecutive)
	{
		ui32_idx_error = ((uint32_t)rand() % ui32_max_dim);
		//printf("\n ui32_idx_error = %u / %u; ui32_n_length = %u", ui32_idx_error, ui32_max_dim, ui32_n_length);
		for (ui32_idx = 0u; ui32_idx < ui32_n_length; ui32_idx++)
		{
			if ((ui32_idx_error + ui32_idx) < ui32_max_dim)
			{
				//printf("fi-");
				pui32_m[ui32_idx_error + ui32_idx] = (uint32_t)rand();
			}
		}
	}
	else
	{
		for (ui32_idx = 0u; ui32_idx < ui32_n_length; ui32_idx++)
		{
			//printf("fi-");
			ui32_idx_error = ((uint32_t)rand() % ui32_max_dim);
			//printf("\n ui32_idx_error = %u / %u; ", ui32_idx_error, ui32_max_dim);
			pui32_m[ui32_idx_error] = (uint32_t)rand();
		}
	}
}


static void_t matrix2zeros(float32_t * paf32_matrix, uint32_t ui32_max_rows, uint32_t ui32_max_columns)
{
	uint32_t ui32_idx;

	for (ui32_idx = 0u; ui32_idx < (ui32_max_rows * ui32_max_columns); ui32_idx++)
	{
		*paf32_matrix++ = 0.0f;
	}
}

static void_t matrix2rand(float32_t * paf32_matrix, uint32_t ui32_max_rows, uint32_t ui32_max_columns)
{
	uint32_t ui32_idx;

	for (ui32_idx = 0u; ui32_idx < (ui32_max_rows * ui32_max_columns); ui32_idx++)
	{
		*paf32_matrix++ = (float32_t)rand();
	}
}

static inline uint32_t singletable_crc32c_ui32(uint32_t ui32_crc, uint32_t ui32_data)
{
	ui32_to_ui8_t u;
	u.ui32 = ui32_data;

	/* 4 bytes*/
	ui32_crc = kaui32_crc_table[(ui32_crc ^ u.ui8[0u]) & 0x00ffu] ^ (ui32_crc >> 8u);
	ui32_crc = kaui32_crc_table[(ui32_crc ^ u.ui8[1u]) & 0x00ffu] ^ (ui32_crc >> 8u);
	ui32_crc = kaui32_crc_table[(ui32_crc ^ u.ui8[2u]) & 0x00ffu] ^ (ui32_crc >> 8u);
	ui32_crc = kaui32_crc_table[(ui32_crc ^ u.ui8[3u]) & 0x00ffu] ^ (ui32_crc >> 8u);

	return ui32_crc;
}

/*==============================================================================================================
* 											Experiment 0
==============================================================================================================*/

/* ==============================================================================================================
* 	Name: smm_no_dc
* ============================================================================================================== */
static uint32_t smm_no_dc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i,
		ui32_idx_j,
		ui32_idx_k;

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++)
		{
			PUT_IN_REGISTER float32_t A_PART = f32_alpha * paf32_ma[(ui32_idx_i * ui32_k) + ui32_idx_k];
			for (ui32_idx_j = 0u; ui32_idx_j < ui32_n; ui32_idx_j++)
			{
				paf32_mc[(ui32_idx_i * ui32_n) + ui32_idx_j] += A_PART * paf32_mb[(ui32_idx_k * ui32_n) + ui32_idx_j];
			}
		}
	}
	return 0u;
}

/* ==============================================================================================================
* 	Name: smm_no_dc_opt
* ============================================================================================================== */
static uint32_t smm_no_dc_opt(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	return 0u;
}

/*==============================================================================================================
* 								 Experiment 1: Individual checksum
==============================================================================================================*/

/*==============================================================================================================
**									Name: smm_xor_external
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the external loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_xor	Return the Execution signature of the MMM
==============================================================================================================*/

static uint32_t smm_xor_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	/* XOR checksum */
	uint32_t ui32_xor_a = 0u,
		ui32_xor_b = 0u,
		ui32_xor_c = 0u,
		ui32_xor;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
		ui32_xor_a ^= (uint32_t) *((uint32_t*)&f32_a_part);
		ui32_xor_b ^= (uint32_t) *((uint32_t*)&f32_b);
		ui32_xor_c ^= (uint32_t) *((uint32_t*)&f32_c);
	}
	ui32_xor = (ui32_xor_a ^ ui32_xor_b) ^ ui32_xor_c;
	return ui32_xor;
}

/*==============================================================================================================
**									Name: smm_xor_intermediate
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_xor	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_xor_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	/* XOR checksum */
	uint32_t ui32_xor_a = 0u,
		ui32_xor_b = 0u,
		ui32_xor_c = 0u,
		ui32_xor;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			ui32_xor_a ^= (uint32_t) *((uint32_t*)&f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
			}
			/* XOR checksum */
			ui32_xor_b ^= (uint32_t) *((uint32_t*)&f32_b);
			ui32_xor_c ^= (uint32_t) *((uint32_t*)&f32_c);

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	ui32_xor = (ui32_xor_a ^ ui32_xor_b) ^ ui32_xor_c;
	return ui32_xor;
}


/*==============================================================================================================
**									Name: smm_xor_internal
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the internal loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_xor	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_xor_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	/* XOR checksum */
	uint32_t ui32_xor_a = 0u,
		ui32_xor_b = 0u,
		ui32_xor_c = 0u,
		ui32_xor;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			ui32_xor_a ^= (uint32_t) *((uint32_t*)&f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];

				/* XOR checksum */
				ui32_xor_b ^= (uint32_t) *((uint32_t*)&f32_b);
				ui32_xor_c ^= (uint32_t) *((uint32_t*)&f32_c);
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	ui32_xor = (ui32_xor_a ^ ui32_xor_b) ^ ui32_xor_c;
	return ui32_xor;
}

/*==============================================================================================================
**									Name: smm_one_external
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with One's checksum in the external loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_xor	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_ones_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	/* One's complement checksum */
	ui64_to_ui32_t Ones_Checksum_a,
		Ones_Checksum_b,
		Ones_Checksum_c,
		Ones_Checksum;

	Ones_Checksum_a.ui64 = 0u;
	Ones_Checksum_b.ui64 = 0u;
	Ones_Checksum_c.ui64 = 0u;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
			}
			ui32_idx_b_ref += ui32_n;
		}

		/* One's complement checksum */
		Ones_Checksum_a.ui64 += (uint64_t) * ((uint32_t*)&f32_a_part);
		Ones_Checksum_a.ui32[0] += Ones_Checksum_a.ui32[1];
		Ones_Checksum_a.ui32[0] = ~Ones_Checksum_a.ui32[0];

		Ones_Checksum_b.ui64 += (uint64_t) * ((uint32_t*)&f32_b);
		Ones_Checksum_b.ui32[0] += Ones_Checksum_b.ui32[1];
		Ones_Checksum_b.ui32[0] = ~Ones_Checksum_b.ui32[0];

		Ones_Checksum_c.ui64 += (uint64_t) * ((uint32_t*)&f32_c);
		Ones_Checksum_c.ui32[0] += Ones_Checksum_c.ui32[1];
		Ones_Checksum_c.ui32[0] = ~Ones_Checksum_c.ui32[0];

		ui32_idx_c_ref += ui32_n;
	}
	Ones_Checksum.ui64 = (Ones_Checksum_a.ui64 + Ones_Checksum_b.ui64) + Ones_Checksum_c.ui64;
	Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
	Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];
	return Ones_Checksum.ui32[0];
}

/*==============================================================================================================
**									Name: smm_one_intermediate
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with One's checksum in the intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_xor	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_ones_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	/* One's complement checksum */
	ui64_to_ui32_t Ones_Checksum_a,
		Ones_Checksum_b,
		Ones_Checksum_c,
		Ones_Checksum;

	Ones_Checksum_a.ui64 = 0u;
	Ones_Checksum_b.ui64 = 0u;
	Ones_Checksum_c.ui64 = 0u;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
			}
			/* One's complement checksum */
			Ones_Checksum_a.ui64 += (uint64_t) * ((uint32_t*)&f32_a_part);
			Ones_Checksum_a.ui32[0] += Ones_Checksum_a.ui32[1];
			Ones_Checksum_a.ui32[0] = ~Ones_Checksum_a.ui32[0];

			Ones_Checksum_b.ui64 += (uint64_t) * ((uint32_t*)&f32_b);
			Ones_Checksum_b.ui32[0] += Ones_Checksum_b.ui32[1];
			Ones_Checksum_b.ui32[0] = ~Ones_Checksum_b.ui32[0];

			Ones_Checksum_c.ui64 += (uint64_t) * ((uint32_t*)&f32_c);
			Ones_Checksum_c.ui32[0] += Ones_Checksum_c.ui32[1];
			Ones_Checksum_c.ui32[0] = ~Ones_Checksum_c.ui32[0];

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	Ones_Checksum.ui64 = (Ones_Checksum_a.ui64 + Ones_Checksum_b.ui64) + Ones_Checksum_c.ui64;
	Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
	Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];
	return Ones_Checksum.ui32[0];
}


/*==============================================================================================================
**									Name: smm_ones_internal
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with One's checksum in the internal loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_xor	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_ones_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	/* One's complement checksum */
	ui64_to_ui32_t Ones_Checksum_a,
		Ones_Checksum_b,
		Ones_Checksum_c,
		Ones_Checksum;

	Ones_Checksum_a.ui64 = 0u;
	Ones_Checksum_b.ui64 = 0u;
	Ones_Checksum_c.ui64 = 0u;


	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			Ones_Checksum_a.ui64 += (uint64_t) * ((uint32_t*)&f32_a_part);
			Ones_Checksum_a.ui32[0] += Ones_Checksum_a.ui32[1];
			Ones_Checksum_a.ui32[0] = ~Ones_Checksum_a.ui32[0];

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n;
				ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
				/* One's complement checksum */
				Ones_Checksum_b.ui64 += (uint64_t) * ((uint32_t*)&f32_b);
				Ones_Checksum_b.ui32[0] += Ones_Checksum_b.ui32[1];
				Ones_Checksum_b.ui32[0] = ~Ones_Checksum_b.ui32[0];

				Ones_Checksum_c.ui64 += (uint64_t) * ((uint32_t*)&f32_c);
				Ones_Checksum_c.ui32[0] += Ones_Checksum_c.ui32[1];
				Ones_Checksum_c.ui32[0] = ~Ones_Checksum_c.ui32[0];
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	Ones_Checksum.ui64 = (Ones_Checksum_a.ui64 + Ones_Checksum_b.ui64);
	Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
	Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];
	Ones_Checksum.ui64 += Ones_Checksum_c.ui64;
	Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
	Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];
	return Ones_Checksum.ui32[0];
}

/*==============================================================================================================
**									Name: smm_twos_external
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with Two's checksum in the external loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_xor	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_twos_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	/* Two's complement checksum */
	uint32_t Twos_Checksum_a = 0u,
		Twos_Checksum_b = 0u,
		Twos_Checksum_c = 0u,
		Twos_Checksum;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
			}
			ui32_idx_b_ref += ui32_n;
		}
		/* Two's complement checksum */
		Twos_Checksum_a += (uint32_t) * (uint32_t*)&f32_a_part;
		Twos_Checksum_a = (~Twos_Checksum_a) + 1;
		Twos_Checksum_b += (uint32_t) * (uint32_t*)&f32_b;
		Twos_Checksum_b = (~Twos_Checksum_b) + 1;
		Twos_Checksum_c += (uint32_t) * (uint32_t*)&f32_c;
		Twos_Checksum_c = (~Twos_Checksum_c) + 1;

		ui32_idx_c_ref += ui32_n;
	}
	Twos_Checksum = (Twos_Checksum_a + Twos_Checksum_b) + Twos_Checksum_c;
	Twos_Checksum = (~Twos_Checksum) + 1;

	return Twos_Checksum;
}

/*==============================================================================================================
**									Name: smm_twos_intermediate
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with Two's checksum in the intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_xor	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_twos_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	/* Two's complement checksum */
	uint32_t Twos_Checksum_a = 0u,
		Twos_Checksum_b = 0u,
		Twos_Checksum_c = 0u,
		Twos_Checksum;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
			}
			/* Two's complement checksum */
			Twos_Checksum_a += (uint32_t) * (uint32_t*)&f32_a_part;
			Twos_Checksum_a = (~Twos_Checksum_a) + 1;

			Twos_Checksum_b += (uint32_t) * (uint32_t*)&f32_b;
			Twos_Checksum_b = (~Twos_Checksum_b) + 1;

			Twos_Checksum_c += (uint32_t) * (uint32_t*)&f32_c;
			Twos_Checksum_c = (~Twos_Checksum_c) + 1;

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	Twos_Checksum = (Twos_Checksum_a + Twos_Checksum_b) + Twos_Checksum_c;
	Twos_Checksum = (~Twos_Checksum) + 1;

	return Twos_Checksum;
}


/* ==============================================================================================================
* 	Name: smm_twos_internal
* ============================================================================================================== */
/*!
** @brief Matrix-matrix multiplication (MMM) with Two's checksum in the internal loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_xor	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_twos_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	/* Two's complement checksum */
	uint32_t Twos_Checksum_a = 0u,
		Twos_Checksum_b = 0u,
		Twos_Checksum_c = 0u,
		Twos_Checksum;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			Twos_Checksum_a += (uint32_t) * (uint32_t*)&f32_a_part;
			Twos_Checksum_a = (~Twos_Checksum_a) + 1;

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];

				/* Two's complement checksum */
				Twos_Checksum_b += (uint32_t) * (uint32_t*)&f32_b;
				Twos_Checksum_b = (~Twos_Checksum_b) + 1;

				Twos_Checksum_c += (uint32_t) * (uint32_t*)&f32_c;
				Twos_Checksum_c = (~Twos_Checksum_c) + 1;
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	Twos_Checksum = (Twos_Checksum_a + Twos_Checksum_b);
	Twos_Checksum = (~Twos_Checksum) + 1;
	Twos_Checksum = (Twos_Checksum + Twos_Checksum_c);
	Twos_Checksum = (~Twos_Checksum) + 1;
	return Twos_Checksum;
}


/* ==============================================================================================================
* 	Name: smm_fletcher_external
* ============================================================================================================== */
/*!
** @brief Matrix-matrix multiplication (MMM) with Fletcher checksum in the external loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	Fletcher.ui32	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_fletcher_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	/* Fletcher checksum */
	ui32_to_ui16_t Fletcher_a,
		Fletcher_b,
		Fletcher_c,
		Fletcher;

	Fletcher_a.ui32 = 0u;
	Fletcher_b.ui32 = 0u;
	Fletcher_c.ui32 = 0u;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
			}
			ui32_idx_b_ref += ui32_n;
		}
		/* Fletcher */
		Fletcher_a.ui32 = Fletcher32c_ui32(Fletcher_a, (uint32_t) * (uint32_t*)&f32_a_part);
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, (uint32_t) * (uint32_t*)&f32_b);
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, (uint32_t) * (uint32_t*)&f32_c);

		ui32_idx_c_ref += ui32_n;
	}
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher_a, Fletcher_b.ui32);
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher, Fletcher_c.ui32);
	return Fletcher.ui32;
}

/* ==============================================================================================================
* 	Name: smm_fletcher_intermediate
* ============================================================================================================== */
/*!
** @brief Matrix-matrix multiplication (MMM) with Fletcher checksum in the intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	Fletcher.ui32	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_fletcher_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	/* Fletcher checksum */
	ui32_to_ui16_t Fletcher_a,
		Fletcher_b,
		Fletcher_c,
		Fletcher;

	Fletcher_a.ui32 = 0u;
	Fletcher_b.ui32 = 0u;
	Fletcher_c.ui32 = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
			}
			/* Fletcher */
			Fletcher_a.ui32 = Fletcher32c_ui32(Fletcher_a, (uint32_t) * (uint32_t*)&f32_a_part);
			Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, (uint32_t) * (uint32_t*)&f32_b);
			Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, (uint32_t) * (uint32_t*)&f32_c);

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher_a, Fletcher_b.ui32);
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher, Fletcher_c.ui32);

	return Fletcher.ui32;
}


/* ==============================================================================================================
* 	Name: smm_fletcher_internal
* ============================================================================================================== */
/*!
** @brief Matrix-matrix multiplication (MMM) with Fletcher checksum in the internal loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	Fletcher.ui32	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_fletcher_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	/* Fletcher checksum */
	ui32_to_ui16_t Fletcher_a,
		Fletcher_b,
		Fletcher_c,
		Fletcher;

	Fletcher_a.ui32 = 0u;
	Fletcher_b.ui32 = 0u;
	Fletcher_c.ui32 = 0u;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			/* Fletcher */
			Fletcher_a.ui32 = Fletcher32c_ui32(Fletcher_a, (uint32_t) * (uint32_t*)&f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
				/* Fletcher */
				Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, (uint32_t) * (uint32_t*)&f32_b);
				Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, (uint32_t) * (uint32_t*)&f32_c);
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher_a, Fletcher_b.ui32);
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher, Fletcher_c.ui32);

	return Fletcher.ui32;
}

/* ==============================================================================================================
* 	Name: smm_crc_external
* ============================================================================================================== */
/*!
** @brief Matrix-matrix multiplication (MMM) with CRC checksum in the external loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_crc	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_crc_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t ui32_crc_a = INITIAL_REMAINDER,
		ui32_crc_b = INITIAL_REMAINDER,
		ui32_crc_c = INITIAL_REMAINDER,
		ui32_crc = INITIAL_REMAINDER;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
			}
			ui32_idx_b_ref += ui32_n;
		}
		/* CRC */
		ui32_crc_a = singletable_crc32c_ui32(ui32_crc_a, (uint32_t) * (uint32_t*)&f32_a_part);
		ui32_crc_b = singletable_crc32c_ui32(ui32_crc_b, (uint32_t) * (uint32_t*)&f32_b);
		ui32_crc_c = singletable_crc32c_ui32(ui32_crc_c, (uint32_t) * (uint32_t*)&f32_c);

		ui32_idx_c_ref += ui32_n;
	}
	ui32_crc = singletable_crc32c_ui32(ui32_crc_a, ui32_crc_b);
	ui32_crc = singletable_crc32c_ui32(ui32_crc, ui32_crc_c);
	return ui32_crc;
}

/* ==============================================================================================================
* 	Name: smm_crc_intermediate
* ============================================================================================================== */
/*!
** @brief Matrix-matrix multiplication (MMM) with CRC checksum in the intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_crc	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_crc_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	uint32_t ui32_crc_a = INITIAL_REMAINDER,
		ui32_crc_b = INITIAL_REMAINDER,
		ui32_crc_c = INITIAL_REMAINDER,
		ui32_crc = INITIAL_REMAINDER;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
			}
			/* CRC */
			ui32_crc_a = singletable_crc32c_ui32(ui32_crc_a, (uint32_t) * (uint32_t*)&f32_a_part);
			ui32_crc_b = singletable_crc32c_ui32(ui32_crc_b, (uint32_t) * (uint32_t*)&f32_b);
			ui32_crc_c = singletable_crc32c_ui32(ui32_crc_c, (uint32_t) * (uint32_t*)&f32_c);

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	ui32_crc = singletable_crc32c_ui32(ui32_crc_a, ui32_crc_b);
	ui32_crc = singletable_crc32c_ui32(ui32_crc, ui32_crc_c);
	return ui32_crc;
}


/* ==============================================================================================================
* 	Name: smm_crc_internal
* ============================================================================================================== */
/*!
** @brief Matrix-matrix multiplication (MMM) with CRC checksum in the internal loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_crc	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_crc_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t ui32_crc_a = INITIAL_REMAINDER,
		ui32_crc_b = INITIAL_REMAINDER,
		ui32_crc_c = INITIAL_REMAINDER,
		ui32_crc = INITIAL_REMAINDER;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			ui32_crc_a = singletable_crc32c_ui32(ui32_crc_a, (uint32_t) * (uint32_t*)&f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
				/* CRC */
				ui32_crc_b = singletable_crc32c_ui32(ui32_crc_b, (uint32_t) * (uint32_t*)&f32_b);
				ui32_crc_c = singletable_crc32c_ui32(ui32_crc_c, (uint32_t) * (uint32_t*)&f32_c);
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	ui32_crc = singletable_crc32c_ui32(ui32_crc_a, ui32_crc_b);
	ui32_crc = singletable_crc32c_ui32(ui32_crc, ui32_crc_c);
	return ui32_crc;
}

/*==============================================================================================================
* 											Experiment 2
==============================================================================================================*/
/* ==============================================================================================================
* 	Name: smm_xor_flet (XOR in the internal and Fletcher in the intermediate loop)
* ============================================================================================================== */
static uint32_t smm_xor_flet(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);


	/* XOR checksum */
	uint32_t ui32_xor_a = 0u,
		ui32_xor_b = 0u,
		ui32_xor_c = 0u,
		ui32_xor;

	/* Fletcher checksum */
	ui32_to_ui16_t Fletcher;
	Fletcher.ui32 = 0u;


	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			ui32_xor_a ^= (uint32_t) *((uint32_t*)&f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
				/* XOR checksum */
				ui32_xor_b ^= (uint32_t) *((uint32_t*)&f32_b);
				ui32_xor_c ^= (uint32_t) *((uint32_t*)&f32_c);
			}
			/* XOR checksum */
			ui32_xor = (ui32_xor_a ^ ui32_xor_b) ^ ui32_xor_c;

			/* Fletcher checksum */
			Fletcher.ui32 = Fletcher32c_ui32(Fletcher, ui32_xor);
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	return Fletcher.ui32;
}


/* ==============================================================================================================
* 	Name: smm_xor_flet (XOR in the internal and CRC in the intermediate loop)
* ============================================================================================================== */
static uint32_t smm_xor_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	/* CRC */
	uint32_t ui32_crc = INITIAL_REMAINDER;

	/* XOR checksum */
	uint32_t ui32_xor_a = 0u,
		ui32_xor_b = 0u,
		ui32_xor_c = 0u,
		ui32_xor;


	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			/* XOR checksum */
			ui32_xor_a ^= (uint32_t) *((uint32_t*)&f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];

				/* XOR checksum */
				ui32_xor_b ^= (uint32_t) *((uint32_t*)&f32_b);
				ui32_xor_c ^= (uint32_t) *((uint32_t*)&f32_c);
			}
			/* XOR checksum */
			ui32_xor = (ui32_xor_a ^ ui32_xor_b) ^ ui32_xor_c;

			/* CRC */
			ui32_crc = singletable_crc32c_ui32(ui32_crc, ui32_xor);
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	return ui32_crc;
}


/* ==============================================================================================================
* 	Name: smm_ones_flet
* ============================================================================================================== */
static uint32_t smm_ones_flet(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	/* One's complement checksum */
	ui64_to_ui32_t Ones_Checksum_a,
		Ones_Checksum_b,
		Ones_Checksum_c,
		Ones_Checksum;

	Ones_Checksum_a.ui64 = 0u;
	Ones_Checksum_b.ui64 = 0u;
	Ones_Checksum_c.ui64 = 0u;

	/* Fletcher checksum */
	ui32_to_ui16_t Fletcher;
	Fletcher.ui32 = 0;

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];

			/* One's complement checksum */
			Ones_Checksum_a.ui64 += (uint64_t) * ((uint32_t*)&f32_a_part);
			Ones_Checksum_a.ui32[0] += Ones_Checksum_a.ui32[1];
			Ones_Checksum_a.ui32[0] = ~Ones_Checksum_a.ui32[0];

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];
				/* One's complement checksum */
				Ones_Checksum_b.ui64 += (uint64_t) * ((uint32_t*)&f32_b);
				Ones_Checksum_b.ui32[0] += Ones_Checksum_b.ui32[1];
				Ones_Checksum_b.ui32[0] = ~Ones_Checksum_b.ui32[0];

				Ones_Checksum_c.ui64 += (uint64_t) * ((uint32_t*)&f32_c);
				Ones_Checksum_c.ui32[0] += Ones_Checksum_c.ui32[1];
				Ones_Checksum_c.ui32[0] = ~Ones_Checksum_c.ui32[0];

			}
			/* One's complement checksum */
			Ones_Checksum.ui64 = (Ones_Checksum_a.ui64 + Ones_Checksum_b.ui64) + Ones_Checksum_c.ui64;
			Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
			Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];
			/* Fletcher checksum */
			Fletcher.ui32 = Fletcher32c_ui32(Fletcher, Ones_Checksum.ui32[0]);

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	return Fletcher.ui32;
}



/* ==============================================================================================================
* 	Name: smm_ones_crc
* ============================================================================================================== */
static uint32_t smm_ones_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	/* CRC */
	uint32_t ui32_crc = INITIAL_REMAINDER;

	/* One's complement checksum */
	ui64_to_ui32_t Ones_Checksum_a,
		Ones_Checksum_b,
		Ones_Checksum_c,
		Ones_Checksum;

	Ones_Checksum_a.ui64 = 0u;
	Ones_Checksum_b.ui64 = 0u;
	Ones_Checksum_c.ui64 = 0u;

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];

			/* One's complement checksum */
			Ones_Checksum_a.ui64 += (uint64_t) * ((uint32_t*)&f32_a_part);
			Ones_Checksum_a.ui32[0] += Ones_Checksum_a.ui32[1];
			Ones_Checksum_a.ui32[0] = ~Ones_Checksum_a.ui32[0];

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];

				/* One's complement checksum */
				Ones_Checksum_b.ui64 += (uint64_t) * ((uint32_t*)&f32_b);
				Ones_Checksum_b.ui32[0] += Ones_Checksum_b.ui32[1];
				Ones_Checksum_b.ui32[0] = ~Ones_Checksum_b.ui32[0];

				Ones_Checksum_c.ui64 += (uint64_t) * ((uint32_t*)&f32_c);
				Ones_Checksum_c.ui32[0] += Ones_Checksum_c.ui32[1];
				Ones_Checksum_c.ui32[0] = ~Ones_Checksum_c.ui32[0];
			}
			/* One's complement checksum */
			Ones_Checksum.ui64 = (Ones_Checksum_a.ui64 + Ones_Checksum_b.ui64) + Ones_Checksum_c.ui64;
			Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
			Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];

			/* CRC */
			ui32_crc = singletable_crc32c_ui32(ui32_crc, Ones_Checksum.ui32[0]);

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	return ui32_crc;
}

/* ==============================================================================================================
* 	Name: smm_twos_flet (Two's complement addition in the internal and Fletcher in the intermediate loop)
* ============================================================================================================== */
static uint32_t smm_twos_flet(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	/* Two's complement checksum */
	uint32_t Twos_Checksum_a = 0u,
		Twos_Checksum_b = 0u,
		Twos_Checksum_c = 0u,
		Twos_Checksum;

	/* Fletcher checksum */
	ui32_to_ui16_t Fletcher;
	Fletcher.ui32 = 0;

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];

			/* Two's complement checksum */
			Twos_Checksum_a += (uint32_t) * (uint32_t*)&f32_a_part;
			Twos_Checksum_a = (~Twos_Checksum_a) + 1;

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];

				/* Two's complement checksum */
				Twos_Checksum_b += (uint32_t) * (uint32_t*)&f32_b;
				Twos_Checksum_b = (~Twos_Checksum_b) + 1;
				Twos_Checksum_c += (uint32_t) * (uint32_t*)&f32_c;
				Twos_Checksum_c = (~Twos_Checksum_c) + 1;

			}
			/* Two's complement checksum */
			Twos_Checksum = (Twos_Checksum_a + Twos_Checksum_b) + Twos_Checksum_c;
			Twos_Checksum = (~Twos_Checksum) + 1;

			/* Fletcher checksum */
			Fletcher.ui32 = Fletcher32c_ui32(Fletcher, Twos_Checksum);
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	return Fletcher.ui32;
}

/* ==============================================================================================================
* 	Name: smm_twos_crc (Two's complement addition in the internal and CRC in the internal loop)
* ============================================================================================================== */
static uint32_t smm_twos_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	/* CRC */
	uint32_t ui32_crc = INITIAL_REMAINDER;

	/* Two's complement checksum */
	uint32_t Twos_Checksum_a = 0u,
		Twos_Checksum_b = 0u,
		Twos_Checksum_c = 0u,
		Twos_Checksum;

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			/* Two's complement checksum */
			Twos_Checksum_a += (uint32_t) * (uint32_t*)&f32_a_part;
			Twos_Checksum_a = (~Twos_Checksum_a) + 1;



			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];

				/* Two's complement checksum */
				Twos_Checksum_b += (uint32_t) * (uint32_t*)&f32_b;
				Twos_Checksum_b = (~Twos_Checksum_b) + 1;
				Twos_Checksum_c += (uint32_t) * (uint32_t*)&f32_c;
				Twos_Checksum_c = (~Twos_Checksum_c) + 1;
			}
			/* Two's complement checksum */
			Twos_Checksum = (Twos_Checksum_a + Twos_Checksum_b) + Twos_Checksum_c;
			Twos_Checksum = (~Twos_Checksum) + 1;

			/* CRC */
			ui32_crc = singletable_crc32c_ui32(ui32_crc, Twos_Checksum);

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	return ui32_crc;
}

/* ==============================================================================================================
* 	Name: smm_flet_crc
* ============================================================================================================== */
static uint32_t smm_flet_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	/* CRC */
	uint32_t ui32_crc = INITIAL_REMAINDER;

	/* Fletcher checksum */
	ui32_to_ui16_t Fletcher_a,
		Fletcher_b,
		Fletcher_c,
		Fletcher;

	Fletcher_a.ui32 = 0u;
	Fletcher_b.ui32 = 0u;
	Fletcher_c.ui32 = 0u;

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			/* Fletcher */
			Fletcher_a.ui32 = Fletcher32c_ui32(Fletcher_a, (uint32_t) * (uint32_t*)&f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];

				/* Fletcher */
				Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, (uint32_t) * (uint32_t*)&f32_b);
				Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, (uint32_t) * (uint32_t*)&f32_c);
			}
			Fletcher.ui32 = (Fletcher_a.ui32 ^ Fletcher_b.ui32) ^ Fletcher_c.ui32;

			/* CRC */
			ui32_crc = singletable_crc32c_ui32(ui32_crc, Fletcher.ui32);

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	return ui32_crc;
}

/*==============================================================================================================
* 	CRC    See: web.mit.edu � freebsd � head � sys � libkern � crc32
* ============================================================================================================== */

/*==============================================================================================================
* 							Fletcher
* ============================================================================================================== */
static inline uint32_t Fletcher32c_ui32(ui32_to_ui16_t Fletcher, uint32_t ui32_data)
{
	ui32_to_ui16_t v;
	v.ui32 = ui32_data;

	Fletcher.ui16[0] += v.ui16[0];
	Fletcher.ui16[1] += Fletcher.ui16[0];
	Fletcher.ui16[0] += v.ui16[1];
	Fletcher.ui16[1] += Fletcher.ui16[0];
	Fletcher.ui16[0] %= 255;
	Fletcher.ui16[1] %= 255;

	return Fletcher.ui32;
}

/*==============================================================================================================
* 							Experiment 3 : Additionals cheksums added by JOP
==============================================================================================================*/

/* ==============================================================================================================
* 	Name: smm_crc_intermediate_comb
* ============================================================================================================== */
static uint32_t smm_crc_intermediate_comb(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i,
		ui32_idx_j,
		ui32_idx_k,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_c_ref = 0u;
	float32_t f32_a_part;
	uint32_t  ui32_XOR = 0u;
	uint32_t  ui32_crc = INITIAL_REMAINDER;
	ui32_to_ui8_t u;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		for (ui32_idx_k = 0u, ui32_idx_b = 0u; ui32_idx_k < ui32_k; ui32_idx_k++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a++];
			ui32_XOR ^= *((uint32_t *)&f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				paf32_mc[ui32_idx_c] += (f32_a_part * paf32_mb[ui32_idx_b]);

				ui32_XOR ^= *((uint32_t *)&paf32_mb[ui32_idx_b]);
				ui32_XOR ^= *((uint32_t *)&paf32_mc[ui32_idx_c]);
			}
			/* CRC */
			SINGLETABLE_CRC32_UI32(ui32_crc, ui32_XOR, u);
		}
		ui32_idx_c_ref += ui32_n;
	}

	return ui32_crc;
}


/* ==============================================================================================================
* 	Name: smm_crc_internal
* ============================================================================================================== */
static uint32_t smm_crc_internal_comb(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;
	uint32_t ui32_XOR = 0u;
	uint32_t ui32_crc = INITIAL_REMAINDER;
	ui32_to_ui8_t u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u; ui32_idx_k < ui32_k; ui32_idx_k++, ui32_idx_a++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			ui32_XOR ^= *((uint32_t *)&f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];

				ui32_XOR ^= *((uint32_t *)&f32_b);
				ui32_XOR ^= *((uint32_t *)&f32_c);

				/* CRC */
				SINGLETABLE_CRC32_UI32(ui32_crc, ui32_XOR, u);
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	return ui32_crc;
}

static uint32_t smm_comb(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t* const paf32_ma, const float32_t* const paf32_mb, float32_t* const paf32_mc)
{
	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t ui32_xor_a = 0u,
		ui32_xor_b = 0u,
		ui32_xor_c = 0u,
		ui32_xor;
	uint32_t ui32_crc = INITIAL_REMAINDER;
	ui32_to_ui8_t u;

	float32_t f32_a_part = 0.0f,
		f32_b = 0.0f,
		f32_c = 0.0f;

	// Verification of the input values
	assert(paf32_ma != NULL);
	assert(paf32_mb != NULL);
	assert(paf32_mc != NULL);

	for (ui32_idx_i = 0u; ui32_idx_i < ui32_m; ui32_idx_i++)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0u, ui32_idx_b = 0u; ui32_idx_k < ui32_k; ui32_idx_k++)
		{
			PUT_IN_REGISTER f32_a_part = f32_alpha * paf32_ma[ui32_idx_a++];
			ui32_xor_a ^= *((uint32_t *)&f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j < ui32_n; ui32_idx_j++, ui32_idx_b++, ui32_idx_c++)
			{
				f32_b = paf32_mb[ui32_idx_b];
				paf32_mc[ui32_idx_c] += f32_a_part * f32_b;
				f32_c = paf32_mc[ui32_idx_c];

				ui32_xor_b ^= *((uint32_t *)&f32_b);
				ui32_xor_c ^= *((uint32_t *)&f32_c);
			}
			/* CRC */
			ui32_xor = (ui32_xor_a ^ ui32_xor_b) ^ ui32_xor_c;
			SINGLETABLE_CRC32_UI32(ui32_crc, ui32_xor, u);

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}

	return ui32_crc;
}
/*==============================================================================================================
* 											Experiment 4
==============================================================================================================*/
#ifdef _WIN32
#ifdef _WIN32
//  Windows
#define cpuid(info, x)    __cpuidex(info, x, 0)
#else
//  GCC Intrinsics
void cpuid(int info[4], int InfoType) {
	__cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#endif
#endif // _AVX

#ifdef _WIN32
//  Misc.
static int HW_MMX, HW_x64, HW_RDRAND, HW_BMI1, HW_BMI2, HW_ADX, HW_PREFETCHWT1;
static int HW_ABM;      // Advanced Bit Manipulation

//  SIMD: 128-bit
static int HW_SSE, HW_SSE2, HW_SSE3, HW_SSSE3, HW_SSE41, HW_SSE42, HW_SSE4a, HW_AES, HW_SHA;

//  SIMD: 256-bit
static int HW_AVX, HW_XOP, HW_FMA3, HW_FMA4, HW_AVX2;

//  SIMD: 512-bit
static int HW_AVX512F;    //  AVX512 Foundation
static int HW_AVX512CD;   //  AVX512 Conflict Detection
static int HW_AVX512PF;   //  AVX512 Prefetch
static int HW_AVX512ER;   //  AVX512 Exponential + Reciprocal
static int HW_AVX512VL;   //  AVX512 Vector Length Extensions
static int HW_AVX512BW;   //  AVX512 Byte + Word
static int HW_AVX512DQ;   //  AVX512 Doubleword + Quadword
static int HW_AVX512IFMA; //  AVX512 Integer 52-bit Fused Multiply-Add
static int HW_AVX512VBMI; //  AVX512 Vector Byte Manipulation Instructions


void check_cpu_features(void) {
	int info[4];
	cpuid(info, 0);
	int nIds = info[0];

	cpuid(info, 0x80000000);
	unsigned nExIds = info[0];

	//  Detect Features
	if (nIds >= 0x00000001) {
		cpuid(info, 0x00000001);
		HW_MMX = (info[3] & ((uint32_t)1 << 23)) != 0;
		HW_SSE = (info[3] & ((uint32_t)1 << 25)) != 0;
		HW_SSE2 = (info[3] & ((uint32_t)1 << 26)) != 0;
		HW_SSE3 = (info[2] & ((uint32_t)1 << 0)) != 0;

		HW_SSSE3 = (info[2] & ((uint32_t)1 << 9)) != 0;
		HW_SSE41 = (info[2] & ((uint32_t)1 << 19)) != 0;
		HW_SSE42 = (info[2] & ((uint32_t)1 << 20)) != 0;
		HW_AES = (info[2] & ((uint32_t)1 << 25)) != 0;

		HW_AVX = (info[2] & ((uint32_t)1 << 28)) != 0;
		HW_FMA3 = (info[2] & ((uint32_t)1 << 12)) != 0;

		HW_RDRAND = (info[2] & ((uint32_t)1 << 30)) != 0;
	}
	if (nIds >= 0x00000007) {
		cpuid(info, 0x00000007);
		HW_AVX2 = (info[1] & ((uint32_t)1 << 5)) != 0;

		HW_BMI1 = (info[1] & ((uint32_t)1 << 3)) != 0;
		HW_BMI2 = (info[1] & ((uint32_t)1 << 8)) != 0;
		HW_ADX = (info[1] & ((uint32_t)1 << 19)) != 0;
		HW_SHA = (info[1] & ((uint32_t)1 << 29)) != 0;
		HW_PREFETCHWT1 = (info[2] & ((uint32_t)1 << 0)) != 0;

		HW_AVX512F = (info[1] & ((uint32_t)1 << 16)) != 0;
		HW_AVX512CD = (info[1] & ((uint32_t)1 << 28)) != 0;
		HW_AVX512PF = (info[1] & ((uint32_t)1 << 26)) != 0;
		HW_AVX512ER = (info[1] & ((uint32_t)1 << 27)) != 0;
		HW_AVX512VL = (info[1] & ((uint32_t)1 << 31)) != 0;
		HW_AVX512BW = (info[1] & ((uint32_t)1 << 30)) != 0;
		HW_AVX512DQ = (info[1] & ((uint32_t)1 << 17)) != 0;
		HW_AVX512IFMA = (info[1] & ((uint32_t)1 << 21)) != 0;
		HW_AVX512VBMI = (info[2] & ((uint32_t)1 << 1)) != 0;
	}
	if (nExIds >= 0x80000001) {
		cpuid(info, 0x80000001);
		HW_x64 = (info[3] & ((uint32_t)1 << 29)) != 0;
		HW_ABM = (info[2] & ((uint32_t)1 << 5)) != 0;
		HW_SSE4a = (info[2] & ((uint32_t)1 << 6)) != 0;
		HW_FMA4 = (info[2] & ((uint32_t)1 << 16)) != 0;
		HW_XOP = (info[2] & ((uint32_t)1 << 11)) != 0;
	}
}
int is_avx() {
	static int result = -1;
	if (result == -1) {
		check_cpu_features();
		result = HW_AVX;
		if (result == 1) printf(" Used AVX \n");
		else printf(" Not used AVX \n");
	}

	return result;
}
#endif

/*==============================================================================================================
**									Name: smm_intel_xor_external
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the external loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return __m256  	m256_xor	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_xor_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	/* XOR checksum */
	__m256i m256_xor_a = _mm256_setzero_si256(),
		m256_xor_b = _mm256_setzero_si256(),
		m256_xor_c = _mm256_setzero_si256(),
		m256_xor = _mm256_setzero_si256();

	/* XOR checksum */
	uint32_t ui32_xor_a = 0u,
		ui32_xor_b = 0u,
		ui32_xor_c = 0u,
		ui32_xor = 0u;

	__m256 a256, b256, c256, result256;    // AVX
	float f32_a_part;

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
				}
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_xor ^= (uint32_t) *((uint32_t*)&f32_a_part);
		m256_xor_b = _mm256_xor_si256(m256_xor_b, _mm256_castps_si256(b256));
		m256_xor_c = _mm256_xor_si256(m256_xor_b, _mm256_castps_si256(result256));
		ui32_idx_c_ref += ui32_n;
	}
	m256_xor = _mm256_xor_si256(m256_xor_b, m256_xor_c);

	uint32_t val[8];
	memcpy(val, &m256_xor, sizeof(val));
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 8; ui32_idx_i++)
	{
		ui32_xor ^= (uint32_t) *((uint32_t*)&val[ui32_idx_i]);
	}
	return ui32_xor;

}

/*==============================================================================================================
**									Name: smm_intel_xor_intermediate
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_xor	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_xor_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	/* XOR checksum */
	__m256i m256_xor_b = _mm256_setzero_si256(),
		m256_xor_c = _mm256_setzero_si256(),
		m256_xor;

	/* XOR checksum */
	uint32_t ui32_xor_a = 0u,
		ui32_xor_b = 0u,
		ui32_xor_c = 0u,
		ui32_xor = 0u;

	__m256 a256, b256, c256, result256;    // AVX
	float f32_a_part;

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			// Evaluation of the ES (A value)
			ui32_xor_a ^= (uint32_t) *((uint32_t*)&f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);
			}

			// Evaluation of the ES (B value)
			m256_xor_b = _mm256_xor_si256(m256_xor_b, _mm256_castps_si256(b256));

			// Evaluation of the ES (C value)
			m256_xor_c = _mm256_xor_si256(m256_xor_c, _mm256_castps_si256(result256));

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
					/* XOR checksum */
					ui32_xor_b ^= (uint32_t) *((uint32_t*)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					ui32_xor_c ^= (uint32_t) *((uint32_t*)&paf32_mc[ui32_idx_c + ui32_idx_j]);
				}
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	ui32_xor ^= (ui32_xor_b ^ ui32_xor_a) ^ ui32_xor_c;

	m256_xor = _mm256_xor_si256(m256_xor_b, m256_xor_c);

	uint32_t val[8];
	memcpy(val, &m256_xor, sizeof(val));
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 8; ui32_idx_i++)
	{
		ui32_xor ^= (uint32_t) *((uint32_t*)&val[ui32_idx_i]);
	}

	return ui32_xor;
}

/*==============================================================================================================
**									Name: smm_intel_xor_internal
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the internal loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 		Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 		Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 		Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_xor	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_xor_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	/* XOR checksum */
	__m256i m256_xor_b = _mm256_setzero_si256(),
		m256_xor_c = _mm256_setzero_si256(),
		m256_xor;

	/* XOR checksum */
	uint32_t ui32_xor_a = 0u,
		ui32_xor_b = 0u,
		ui32_xor_c = 0u,
		ui32_xor = 0u;

	__m256 a256, b256, c256, result256;    // AVX
	float f32_a_part;

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			// Evaluation of the ES (A value)
			ui32_xor_a ^= (uint32_t) *((uint32_t*)&f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);

				// Evaluation of the ES (B value)
				m256_xor_b = _mm256_xor_si256(m256_xor_b, _mm256_castps_si256(b256));

				// Evaluation of the ES (C value)
				m256_xor_c = _mm256_xor_si256(m256_xor_c, _mm256_castps_si256(result256));
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
					/* XOR checksum */
					ui32_xor_b ^= (uint32_t) *((uint32_t*)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					ui32_xor_c ^= (uint32_t) *((uint32_t*)&paf32_mc[ui32_idx_c + ui32_idx_j]);
				}
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	ui32_xor ^= (ui32_xor_b ^ ui32_xor_a) ^ ui32_xor_c;

	m256_xor = _mm256_xor_si256(m256_xor_b, m256_xor_c);

	uint32_t val[8];
	memcpy(val, &m256_xor, sizeof(val));
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 8; ui32_idx_i++)
	{
		ui32_xor ^= (uint32_t) *((uint32_t*)&val[ui32_idx_i]);
	}

	return ui32_xor;
}
/*==============================================================================================================
**									Name: smm_intel_twos_external
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the external loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return __m256  	m256_twos	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_twos_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	/* Twos checksum */
	__m256i m256i_twos_b = _mm256_setzero_si256(),
		m256i_twos_c = _mm256_setzero_si256(),
		m256i_twos;

	/* Two's complement checksum */
	uint32_t Twos_Checksum_a = 0u,
		Twos_Checksum_b = 0u,
		Twos_Checksum_c = 0u,
		Twos_Checksum = 0u;

	__m256i m256i_Ones = _mm256_set1_epi32(-1);
	__m256i m256i_singleOne = _mm256_set1_epi32(1);

	__m256 a256, b256, c256, result256;    // AVX
	float f32_a_part;

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
				}
			}
			ui32_idx_b_ref += ui32_n;
		}

		// Evaluation of the ES (A value)
		Twos_Checksum_a += (uint32_t) * (uint32_t*)&f32_a_part;
		Twos_Checksum_a = (~Twos_Checksum_a) + 1;

		// Evaluation of the ES (B value)
		m256i_twos_b = _mm256_add_epi32(m256i_twos_b, _mm256_castps_si256(b256));
		m256i_twos_b = _mm256_xor_si256(m256i_twos_b, m256i_Ones);
		m256i_twos_b = _mm256_add_epi32(m256i_twos_b, m256i_singleOne);

		// Evaluation of the ES (C value)
		m256i_twos_c = _mm256_add_epi32(m256i_twos_c, _mm256_castps_si256(result256));
		m256i_twos_c = _mm256_xor_si256(m256i_twos_c, m256i_Ones);
		m256i_twos_c = _mm256_add_epi32(m256i_twos_c, m256i_singleOne);

		ui32_idx_c_ref += ui32_n;
	}
	Twos_Checksum = Twos_Checksum_a;

	m256i_twos = _mm256_add_epi32(m256i_twos_b, m256i_twos_c);
	m256i_twos = _mm256_xor_si256(m256i_twos, m256i_Ones);
	m256i_twos = _mm256_add_epi32(m256i_twos, m256i_singleOne);

	uint32_t val[8];
	memcpy(val, &m256i_twos, sizeof(val));
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 8; ui32_idx_i++)
	{
		/* Two's complement checksum */
		Twos_Checksum += (uint32_t) * (uint32_t*)&val[ui32_idx_i];
		Twos_Checksum = (~Twos_Checksum) + 1;
	}
	return Twos_Checksum;
}

/*==============================================================================================================
**									Name: smm_intel_twos_intermediate
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_twos	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_twos_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	/* Twos checksum */
	__m256i m256i_twos_b = _mm256_setzero_si256(),
		m256i_twos_c = _mm256_setzero_si256(),
		m256i_twos;

	/* Two's complement checksum */
	uint32_t Twos_Checksum_a = 0u,
		Twos_Checksum_b = 0u,
		Twos_Checksum_c = 0u,
		Twos_Checksum = 0u;

	__m256i m256i_Ones = _mm256_set1_epi32(-1);
	__m256i m256i_singleOne = _mm256_set1_epi32(1);

	__m256 a256, b256, c256, result256;    // AVX
	float f32_a_part;

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
				}
			}

			// Evaluation of the ES (A value)
			Twos_Checksum_a += (uint32_t) * (uint32_t*)&f32_a_part;
			Twos_Checksum_a = (~Twos_Checksum_a) + 1;

			// Evaluation of the ES (B value)
			m256i_twos_b = _mm256_add_epi32(m256i_twos_b, _mm256_castps_si256(b256));
			m256i_twos_b = _mm256_xor_si256(m256i_twos_b, m256i_Ones);
			m256i_twos_b = _mm256_add_epi32(m256i_twos_b, m256i_singleOne);

			// Evaluation of the ES (C value)
			m256i_twos_c = _mm256_add_epi32(m256i_twos_c, _mm256_castps_si256(result256));
			m256i_twos_c = _mm256_xor_si256(m256i_twos_c, m256i_Ones);
			m256i_twos_c = _mm256_add_epi32(m256i_twos_c, m256i_singleOne);

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	Twos_Checksum += Twos_Checksum_b + Twos_Checksum_a;
	Twos_Checksum = (~Twos_Checksum) + 1;
	Twos_Checksum += Twos_Checksum_c;
	Twos_Checksum = (~Twos_Checksum) + 1;

	m256i_twos = _mm256_add_epi32(m256i_twos_b, m256i_twos_c);
	m256i_twos = _mm256_xor_si256(m256i_twos, m256i_Ones);
	m256i_twos = _mm256_add_epi32(m256i_twos, m256i_singleOne);

	uint32_t val[8];
	memcpy(val, &m256i_twos, sizeof(val));
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 8; ui32_idx_i++)
	{
		/* Two's complement checksum */
		Twos_Checksum += val[ui32_idx_i];
		Twos_Checksum = (~Twos_Checksum) + 1;
	}

	return Twos_Checksum;
}

/*==============================================================================================================
**									Name: smm_intel_twos_internal
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the internal loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 		Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 		Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 		Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_twos	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_twos_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	/* Twos checksum */
	__m256i m256i_twos_a= _mm256_setzero_si256(),
		m256i_twos_b = _mm256_setzero_si256(),
		m256i_twos_c = _mm256_setzero_si256(),
		m256i_twos = _mm256_setzero_si256();

	/* Two's complement checksum */
	uint32_t Twos_Checksum_a = 0u,
		Twos_Checksum_b = 0u,
		Twos_Checksum_c = 0u,
		Twos_Checksum = 0u;

	__m256i m256i_Ones = _mm256_set1_epi32(-1);
	__m256i m256i_singleOne = _mm256_set1_epi32(1);

	__m256 a256, b256, c256, result256;    // AVX
	float f32_a_part;

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);
	uint32_t val[8];

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);


			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);

				// Evaluation of the ES (B value)
				m256i_twos_b = _mm256_add_epi32(m256i_twos_b, _mm256_castps_si256(b256));
				m256i_twos_b = _mm256_xor_si256(m256i_twos_b, m256i_Ones);
				m256i_twos_b = _mm256_add_epi32(m256i_twos_b, m256i_singleOne);

				// Evaluation of the ES (C value)
				m256i_twos_c = _mm256_add_epi32(m256i_twos_c, _mm256_castps_si256(result256));
				m256i_twos_c = _mm256_xor_si256(m256i_twos_c, m256i_Ones);
				m256i_twos_c = _mm256_add_epi32(m256i_twos_c, m256i_singleOne);
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
					/* Twos checksum */
					// Evaluation of the ES (B value)
					Twos_Checksum_b += (uint32_t) *((uint32_t*)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					Twos_Checksum_b = (~Twos_Checksum_b) + 1;
					// Evaluation of the ES (C value)
					Twos_Checksum_c += (uint32_t) *((uint32_t*)&paf32_mc[ui32_idx_c + ui32_idx_j]);
					Twos_Checksum_c = (~Twos_Checksum_c) + 1;
				}
			}
			// Evaluation of the ES (A value)
			Twos_Checksum_a += (uint32_t) * (uint32_t*)&f32_a_part;
			Twos_Checksum_a = (~Twos_Checksum_a) + 1;

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	if (0 != prev_end)
	{
		Twos_Checksum = Twos_Checksum_b + Twos_Checksum_a;
		Twos_Checksum = (~Twos_Checksum) + 1;
		Twos_Checksum += Twos_Checksum_c;
		Twos_Checksum = (~Twos_Checksum) + 1;
	}
	else
	{
		Twos_Checksum = Twos_Checksum_a;
	}
	m256i_twos = _mm256_add_epi32(m256i_twos_b, m256i_twos_c);
	m256i_twos = _mm256_xor_si256(m256i_twos, m256i_Ones);
	m256i_twos = _mm256_add_epi32(m256i_twos, m256i_singleOne);

	memcpy(val, &m256i_twos, sizeof(val));
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 8; ui32_idx_i++)
	{
		/* Two's complement checksum */
		Twos_Checksum += val[ui32_idx_i];
		Twos_Checksum = (~Twos_Checksum) + 1;
	}

	return Twos_Checksum;
}

/*==============================================================================================================
**									Name: smm_intel_ones_external
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the external loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return __m256  	m256_ones	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_ones_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	// Variables to store the ES
	__m256i  Ones_checksum_b_hi = _mm256_setzero_si256(),
		Ones_checksum_b_lo = _mm256_setzero_si256(),
		Ones_checksum_c_hi = _mm256_setzero_si256(),
		Ones_checksum_c_lo = _mm256_setzero_si256();

	__m256i b_aux_lo, b_aux_hi,
		c_aux_lo, c_aux_hi,
		m256i_zeros = _mm256_setzero_si256();

	__m256i m256i_Ones = _mm256_set1_epi32(-1);
	__m256 a256, b256, c256, result256;    // AVX

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	/* One's complement checksum */
	ui64_to_ui32_t Ones_Checksum_a,
		Ones_Checksum_b,
		Ones_Checksum_c,
		Ones_Checksum;

	Ones_Checksum_a.ui64 = 0u;
	Ones_Checksum_b.ui64 = 0u;
	Ones_Checksum_c.ui64 = 0u;
	Ones_Checksum.ui64 = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	float f32_a_part;
	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);
			}
			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
				}
			}
			ui32_idx_b_ref += ui32_n;
		}
		// Evaluation of the ES (A value)
		Ones_Checksum_a.ui64 += (uint64_t) * ((uint32_t*)&f32_a_part);
		Ones_Checksum_a.ui32[0] += Ones_Checksum_a.ui32[1];
		Ones_Checksum_a.ui32[0] = ~Ones_Checksum_a.ui32[0];

		// Evaluation of the ES (B value)
		b_aux_hi = _mm256_unpackhi_epi32(_mm256_castps_si256(b256), m256i_zeros);
		Ones_checksum_b_hi = _mm256_add_epi64(Ones_checksum_b_hi, b_aux_hi);
		Ones_checksum_b_hi = _mm256_hadd_epi32(Ones_checksum_b_hi, Ones_checksum_b_hi);
		Ones_checksum_b_hi = _mm256_xor_si256(Ones_checksum_b_hi, m256i_Ones);
		Ones_checksum_b_hi = _mm256_unpackhi_epi32(Ones_checksum_b_hi, m256i_zeros);

		b_aux_lo = _mm256_unpacklo_epi32(_mm256_castps_si256(b256), m256i_zeros);
		Ones_checksum_b_lo = _mm256_add_epi64(Ones_checksum_b_lo, b_aux_lo);
		Ones_checksum_b_lo = _mm256_hadd_epi32(Ones_checksum_b_lo, Ones_checksum_b_lo);
		Ones_checksum_b_lo = _mm256_xor_si256(Ones_checksum_b_lo, m256i_Ones);
		Ones_checksum_b_lo = _mm256_unpackhi_epi32(Ones_checksum_b_lo, m256i_zeros);


		// Evaluation of the ES (C value)
		c_aux_hi = _mm256_unpackhi_epi32(_mm256_castps_si256(result256), m256i_zeros);
		Ones_checksum_c_hi = _mm256_add_epi64(Ones_checksum_c_hi, c_aux_hi);
		Ones_checksum_c_hi = _mm256_hadd_epi32(Ones_checksum_c_hi, Ones_checksum_c_hi);
		Ones_checksum_c_hi = _mm256_xor_si256(Ones_checksum_c_hi, m256i_Ones);
		Ones_checksum_c_hi = _mm256_unpackhi_epi32(Ones_checksum_c_hi, m256i_zeros);


		c_aux_lo = _mm256_unpacklo_epi32(_mm256_castps_si256(result256), m256i_zeros);
		Ones_checksum_c_lo = _mm256_add_epi64(Ones_checksum_c_lo, c_aux_lo);
		Ones_checksum_c_lo = _mm256_hadd_epi32(Ones_checksum_c_lo, Ones_checksum_c_lo);
		Ones_checksum_c_lo = _mm256_xor_si256(Ones_checksum_c_lo, m256i_Ones);
		Ones_checksum_c_lo = _mm256_unpackhi_epi32(Ones_checksum_c_lo, m256i_zeros);

		ui32_idx_c_ref += ui32_n;
	}
	Ones_Checksum.ui64 = (Ones_Checksum_a.ui64);

	// Evaluation of the ES (C value)
	Ones_checksum_c_hi = _mm256_add_epi64(Ones_checksum_c_hi, Ones_checksum_b_hi);
	Ones_checksum_c_hi = _mm256_hadd_epi32(Ones_checksum_c_hi, Ones_checksum_c_hi);
	Ones_checksum_c_hi = _mm256_unpackhi_epi32(Ones_checksum_c_hi, m256i_zeros);
	Ones_checksum_c_hi = _mm256_xor_si256(Ones_checksum_c_hi, m256i_Ones);

	Ones_checksum_c_lo = _mm256_add_epi64(Ones_checksum_c_lo, Ones_checksum_b_lo);
	Ones_checksum_c_lo = _mm256_hadd_epi32(Ones_checksum_c_lo, Ones_checksum_c_lo);
	Ones_checksum_c_lo = _mm256_unpackhi_epi32(Ones_checksum_c_lo, m256i_zeros);
	Ones_checksum_c_lo = _mm256_xor_si256(Ones_checksum_c_lo, m256i_Ones);

	uint32_t val[8];
	memcpy(val, &Ones_checksum_c_lo, sizeof(val));
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 8; ui32_idx_i++)
	{
		/* One's complement checksum */
		Ones_Checksum.ui64 += (uint64_t) * ((uint32_t*)&val[ui32_idx_i]);
		Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
		Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];
	}
	memcpy(val, &Ones_checksum_c_hi, sizeof(val));
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 8; ui32_idx_i++)
	{
		/* One's complement checksum */
		Ones_Checksum.ui64 += (uint64_t) * ((uint32_t*)&val[ui32_idx_i]);
		Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
		Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];
	}

	return Ones_Checksum.ui32[0];
}

/*==============================================================================================================
**									Name: smm_intel_ones_intermediate
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_ones	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_ones_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	// Variables to store the ES
	__m256i  Ones_checksum_b_hi = _mm256_setzero_si256(),
		Ones_checksum_b_lo = _mm256_setzero_si256(),
		Ones_checksum_c_hi = _mm256_setzero_si256(),
		Ones_checksum_c_lo = _mm256_setzero_si256();

	__m256i b_aux_lo, b_aux_hi,
		c_aux_lo, c_aux_hi,
		m256i_zeros = _mm256_setzero_si256();

	__m256i m256i_Ones = _mm256_set1_epi32(-1);
	__m256 a256, b256, c256, result256;    // AVX

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	/* One's complement checksum */
	ui64_to_ui32_t Ones_Checksum_a,
		Ones_Checksum_b,
		Ones_Checksum_c,
		Ones_Checksum;

	Ones_Checksum_a.ui64 = 0u;
	Ones_Checksum_b.ui64 = 0u;
	Ones_Checksum_c.ui64 = 0u;
	Ones_Checksum.ui64 = 0u;

	float f32_a_part;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);
			}

			// Evaluation of the ES (B value)
			b_aux_hi = _mm256_unpackhi_epi32(_mm256_castps_si256(b256), m256i_zeros);
			Ones_checksum_b_hi = _mm256_add_epi64(Ones_checksum_b_hi, b_aux_hi);
			Ones_checksum_b_hi = _mm256_hadd_epi32(Ones_checksum_b_hi, Ones_checksum_b_hi);
			Ones_checksum_b_hi = _mm256_xor_si256(Ones_checksum_b_hi, m256i_Ones);
			Ones_checksum_b_hi = _mm256_unpackhi_epi32(Ones_checksum_b_hi, m256i_zeros);


			b_aux_lo = _mm256_unpacklo_epi32(_mm256_castps_si256(b256), m256i_zeros);
			Ones_checksum_b_lo = _mm256_add_epi64(Ones_checksum_b_lo, b_aux_lo);
			Ones_checksum_b_lo = _mm256_hadd_epi32(Ones_checksum_b_lo, Ones_checksum_b_lo);
			Ones_checksum_b_lo = _mm256_xor_si256(Ones_checksum_b_lo, m256i_Ones);
			Ones_checksum_b_lo = _mm256_unpackhi_epi32(Ones_checksum_b_lo, m256i_zeros);


			// Evaluation of the ES (C value)
			c_aux_hi = _mm256_unpackhi_epi32(_mm256_castps_si256(result256), m256i_zeros);
			Ones_checksum_c_hi = _mm256_add_epi64(Ones_checksum_c_hi, c_aux_hi);
			Ones_checksum_c_hi = _mm256_hadd_epi32(Ones_checksum_c_hi, Ones_checksum_c_hi);
			Ones_checksum_c_hi = _mm256_xor_si256(Ones_checksum_c_hi, m256i_Ones);
			Ones_checksum_c_hi = _mm256_unpackhi_epi32(Ones_checksum_c_hi, m256i_zeros);


			c_aux_lo = _mm256_unpacklo_epi32(_mm256_castps_si256(result256), m256i_zeros);
			Ones_checksum_c_lo = _mm256_add_epi64(Ones_checksum_c_lo, c_aux_lo);
			Ones_checksum_c_lo = _mm256_hadd_epi32(Ones_checksum_c_lo, Ones_checksum_c_lo);
			Ones_checksum_c_lo = _mm256_xor_si256(Ones_checksum_c_lo, m256i_Ones);
			Ones_checksum_c_lo = _mm256_unpackhi_epi32(Ones_checksum_c_lo, m256i_zeros);


			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
					/* One's complement checksum */
					Ones_Checksum_b.ui64 += (uint64_t) * ((uint32_t*)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					Ones_Checksum_b.ui32[0] += Ones_Checksum_b.ui32[1];
					Ones_Checksum_b.ui32[0] = ~Ones_Checksum_b.ui32[0];

					Ones_Checksum_c.ui64 += (uint64_t) * ((uint32_t*)&paf32_mc[ui32_idx_c + ui32_idx_j]);
					Ones_Checksum_c.ui32[0] += Ones_Checksum_c.ui32[1];
					Ones_Checksum_c.ui32[0] = ~Ones_Checksum_c.ui32[0];
				}
			}
			// Evaluation of the ES (A value)
			Ones_Checksum_a.ui64 += (uint64_t) * ((uint32_t*)&f32_a_part);
			Ones_Checksum_a.ui32[0] += Ones_Checksum_a.ui32[1];
			Ones_Checksum_a.ui32[0] = ~Ones_Checksum_a.ui32[0];

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	if (0 != prev_end) {
		Ones_Checksum.ui64 = (Ones_Checksum_a.ui64 + Ones_Checksum_b.ui64);
		Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
		Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];

		Ones_Checksum.ui64 += Ones_Checksum_c.ui64;
		Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
		Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];
	}
	else {
		Ones_Checksum.ui64 = (Ones_Checksum_a.ui64);
	}

	// Evaluation of the ES (C value)
	Ones_checksum_c_hi = _mm256_add_epi64(Ones_checksum_c_hi, Ones_checksum_b_hi);
	Ones_checksum_c_hi = _mm256_hadd_epi32(Ones_checksum_c_hi, Ones_checksum_c_hi);
	Ones_checksum_c_hi = _mm256_unpackhi_epi32(Ones_checksum_c_hi, m256i_zeros);
	Ones_checksum_c_hi = _mm256_xor_si256(Ones_checksum_c_hi, m256i_Ones);

	Ones_checksum_c_lo = _mm256_add_epi64(Ones_checksum_c_lo, Ones_checksum_b_lo);
	Ones_checksum_c_lo = _mm256_hadd_epi32(Ones_checksum_c_lo, Ones_checksum_c_lo);
	Ones_checksum_c_lo = _mm256_unpackhi_epi32(Ones_checksum_c_lo, m256i_zeros);
	Ones_checksum_c_lo = _mm256_xor_si256(Ones_checksum_c_lo, m256i_Ones);

	uint32_t val[8];
	memcpy(val, &Ones_checksum_c_lo, sizeof(val));
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 8; ui32_idx_i++)
	{
		/* One's complement checksum */
		Ones_Checksum.ui64 += (uint64_t) * ((uint32_t*)&val[ui32_idx_i]);
		Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
		Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];
	}
	memcpy(val, &Ones_checksum_c_hi, sizeof(val));
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 8; ui32_idx_i++)
	{
		/* One's complement checksum */
		Ones_Checksum.ui64 += (uint64_t) * ((uint32_t*)&val[ui32_idx_i]);
		Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
		Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];
	}

	return Ones_Checksum.ui32[0];
}

/*==============================================================================================================
**									Name: smm_intel_ones_internal
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the internal loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 		Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 		Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 		Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_ones	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_ones_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	// Variables to store the ES
	__m256i  Ones_checksum_a_hi = _mm256_setzero_si256(),
		Ones_checksum_a_lo = _mm256_setzero_si256(),
		Ones_checksum_b_hi = _mm256_setzero_si256(),
		Ones_checksum_b_lo = _mm256_setzero_si256(),
		Ones_checksum_c_hi = _mm256_setzero_si256(),
		Ones_checksum_c_lo = _mm256_setzero_si256(),
		Ones_checksum = _mm256_setzero_si256();

	__m256i a_aux_lo, a_aux_hi,
		b_aux_lo, b_aux_hi,
		c_aux_lo, c_aux_hi,
		m256i_zeros = _mm256_setzero_si256();

	__m256i m256i_Ones = _mm256_set1_epi32(-1);
	__m256 a256, b256, c256, result256;    // AVX

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	/* One's complement checksum */
	ui64_to_ui32_t Ones_Checksum_a,
		Ones_Checksum_b,
		Ones_Checksum_c,
		Ones_Checksum;

	Ones_Checksum_a.ui64 = 0u;
	Ones_Checksum_b.ui64 = 0u;
	Ones_Checksum_c.ui64 = 0u;
	Ones_Checksum.ui64 = 0u;

	float f32_a_part;
	uint32_t val[8];
	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);

				// Evaluation of the ES (B value)
				b_aux_hi = _mm256_unpackhi_epi32(_mm256_castps_si256(b256), m256i_zeros);
				Ones_checksum_b_hi = _mm256_add_epi64(Ones_checksum_b_hi, b_aux_hi);
				Ones_checksum_b_hi = _mm256_hadd_epi32(Ones_checksum_b_hi, Ones_checksum_b_hi);
				Ones_checksum_b_hi = _mm256_xor_si256(Ones_checksum_b_hi, m256i_Ones);
				Ones_checksum_b_hi = _mm256_unpackhi_epi32(Ones_checksum_b_hi, m256i_zeros);


				b_aux_lo = _mm256_unpacklo_epi32(_mm256_castps_si256(b256), m256i_zeros);
				Ones_checksum_b_lo = _mm256_add_epi64(Ones_checksum_b_lo, b_aux_lo);
				Ones_checksum_b_lo = _mm256_hadd_epi32(Ones_checksum_b_lo, Ones_checksum_b_lo);
				Ones_checksum_b_lo = _mm256_xor_si256(Ones_checksum_b_lo, m256i_Ones);
				Ones_checksum_b_lo = _mm256_unpackhi_epi32(Ones_checksum_b_lo, m256i_zeros);


				// Evaluation of the ES (C value)
				c_aux_hi = _mm256_unpackhi_epi32(_mm256_castps_si256(result256), m256i_zeros);
				Ones_checksum_c_hi = _mm256_add_epi64(Ones_checksum_c_hi, c_aux_hi);
				Ones_checksum_c_hi = _mm256_hadd_epi32(Ones_checksum_c_hi, Ones_checksum_c_hi);
				Ones_checksum_c_hi = _mm256_xor_si256(Ones_checksum_c_hi, m256i_Ones);
				Ones_checksum_c_hi = _mm256_unpackhi_epi32(Ones_checksum_c_hi, m256i_zeros);


				c_aux_lo = _mm256_unpacklo_epi32(_mm256_castps_si256(result256), m256i_zeros);
				Ones_checksum_c_lo = _mm256_add_epi64(Ones_checksum_c_lo, c_aux_lo);
				Ones_checksum_c_lo = _mm256_hadd_epi32(Ones_checksum_c_lo, Ones_checksum_c_lo);
				Ones_checksum_c_lo = _mm256_xor_si256(Ones_checksum_c_lo, m256i_Ones);
				Ones_checksum_c_lo = _mm256_unpackhi_epi32(Ones_checksum_c_lo, m256i_zeros);
			}

			// Evaluation of the ES (A value)
			/*a_aux_hi = _mm256_unpackhi_epi32(_mm256_castps_si256(a256), m256i_zeros);
			Ones_checksum_a_hi = _mm256_add_epi64(Ones_checksum_a_hi, a_aux_hi);
			Ones_checksum_a_hi = _mm256_hadd_epi32(Ones_checksum_a_hi, Ones_checksum_a_hi);
			Ones_checksum_a_hi = _mm256_xor_si256(Ones_checksum_a_hi, m256i_Ones);
			Ones_checksum_a_hi = _mm256_unpackhi_epi32(Ones_checksum_a_hi, m256i_zeros);

			a_aux_lo = _mm256_unpacklo_epi32(_mm256_castps_si256(a256), m256i_zeros);
			Ones_checksum_a_lo = _mm256_add_epi64(Ones_checksum_a_lo, a_aux_lo);
			Ones_checksum_a_lo = _mm256_hadd_epi32(Ones_checksum_a_lo, Ones_checksum_a_lo);
			Ones_checksum_a_lo = _mm256_xor_si256(Ones_checksum_a_lo, m256i_Ones);
			Ones_checksum_a_lo = _mm256_unpackhi_epi32(Ones_checksum_a_lo, m256i_zeros);
			*/

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
					/* One's complement checksum */
					Ones_Checksum_b.ui64 += (uint64_t) * ((uint32_t*)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					Ones_Checksum_b.ui32[0] += Ones_Checksum_b.ui32[1];
					Ones_Checksum_b.ui32[0] = ~Ones_Checksum_b.ui32[0];

					Ones_Checksum_c.ui64 += (uint64_t) * ((uint32_t*)&paf32_mc[ui32_idx_c + ui32_idx_j]);
					Ones_Checksum_c.ui32[0] += Ones_Checksum_c.ui32[1];
					Ones_Checksum_c.ui32[0] = ~Ones_Checksum_c.ui32[0];
				}
			}
			Ones_Checksum_a.ui64 += (uint64_t) * ((uint32_t*)&f32_a_part);
			Ones_Checksum_a.ui32[0] += Ones_Checksum_a.ui32[1];
			Ones_Checksum_a.ui32[0] = ~Ones_Checksum_a.ui32[0];

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	if (0 != prev_end) {
		Ones_Checksum.ui64 = (Ones_Checksum_a.ui64 + Ones_Checksum_b.ui64);
		Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
		Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];
		Ones_Checksum.ui64 += Ones_Checksum_c.ui64;
		Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
		Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];
	}
	else {
		Ones_Checksum.ui64 = Ones_Checksum_a.ui64;
	}

	// Evaluation of the ES (C value)
	Ones_checksum_b_hi = _mm256_add_epi64(Ones_checksum_b_hi, Ones_checksum_b_lo);
	Ones_checksum_c_hi = _mm256_add_epi64(Ones_checksum_c_hi, Ones_checksum_c_lo);
	Ones_checksum      = _mm256_add_epi64(Ones_checksum_b_hi, Ones_checksum_c_hi);

	Ones_checksum = _mm256_hadd_epi32(Ones_checksum, Ones_checksum);
	Ones_checksum = _mm256_xor_si256(Ones_checksum, m256i_Ones);
	Ones_checksum = _mm256_unpackhi_epi32(Ones_checksum, m256i_zeros);


	memcpy(val, &Ones_checksum, sizeof(val));
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 8; (ui32_idx_i += 2) )
	{
		/* One's complement checksum */
		Ones_Checksum.ui64 += (uint64_t) * ((uint32_t*)&val[ui32_idx_i]);
		Ones_Checksum.ui32[0] += Ones_Checksum.ui32[1];
		Ones_Checksum.ui32[0] = ~Ones_Checksum.ui32[0];
	}

	return Ones_Checksum.ui32[0];
}

/*==============================================================================================================
**									Name: smm_intel_fletcher_external
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the external loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return __m256  	m256_xor	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_fletcher_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	// Fletcher Checksum with SIMD instructions(B and C)
	__m128i m128i_Fletcher_b_lo = _mm_setzero_si128(),
		m128i_Fletcher_b_hi = _mm_setzero_si128(),
		m128i_Fletcher_c_lo = _mm_setzero_si128(),
		m128i_Fletcher_c_hi = _mm_setzero_si128(),
		aux_128i_b_lo, aux_128i_b_hi,
		aux_128i_c_lo, aux_128i_c_hi;

	uint32_t val_b_lo[4] = { 0u },
		val_b_hi[4] = { 0u },
		val_c_lo[4] = { 0u },
		val_c_hi[4] = { 0u };

	/* Fletcher checksum sequential */
	ui32_to_ui16_t Fletcher_a, Fletcher_b, Fletcher_c, Fletcher;
	Fletcher_a.ui32 = 0u;
	Fletcher_b.ui32 = 0u;
	Fletcher_c.ui32 = 0u;
	Fletcher.ui32 = 0u;

	__m256i aux_256i_b, aux_256i_c;
	__m256 a256, b256, c256, result256;    // AVX

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float f32_a_part;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);
			}
			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
				}
			}
			ui32_idx_b_ref += ui32_n;
		}
		// Evaluation of the ES (A value)
		Fletcher_a.ui32 = Fletcher32c_ui32(Fletcher_a, (uint32_t) * (uint32_t*)&f32_a_part);

		// Casting of the variables (B and C values)
		aux_256i_b = _mm256_castps_si256(b256);
		aux_256i_c = _mm256_castps_si256(result256);

		// Evaluation of the ES (A value)
		Fletcher_a.ui32 = Fletcher32c_ui32(Fletcher_a, (uint32_t) * (uint32_t*)&f32_a_part);

		// Fletcher computation B
		aux_128i_b_lo = _mm256_extractf128_si256(aux_256i_b, 0);
		aux_128i_b_hi = _mm256_extractf128_si256(aux_256i_b, 1);

		m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_lo);
		m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);
		m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_hi);
		m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);


		// Fletcher computation C
		aux_128i_c_lo = _mm256_extractf128_si256(aux_256i_c, 0);
		aux_128i_c_hi = _mm256_extractf128_si256(aux_256i_c, 1);

		m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_lo);
		m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);
		m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_hi);
		m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);

		memcpy(val_b_lo, &m128i_Fletcher_b_lo, sizeof(val_b_lo));
		memcpy(val_b_hi, &m128i_Fletcher_b_hi, sizeof(val_b_hi));
		memcpy(val_c_lo, &m128i_Fletcher_c_lo, sizeof(val_c_lo));
		memcpy(val_c_hi, &m128i_Fletcher_c_hi, sizeof(val_c_hi));
		for (uint32_t ui32_idx_l = 0; ui32_idx_l < 4; ui32_idx_l++)
		{
			val_b_lo[ui32_idx_l] %= 65535;
			val_b_hi[ui32_idx_l] %= 65535;
			val_c_lo[ui32_idx_l] %= 65535;
			val_c_hi[ui32_idx_l] %= 65535;
		}
		memcpy(&m128i_Fletcher_b_lo, val_b_lo, sizeof(val_b_lo));
		memcpy(&m128i_Fletcher_b_hi, val_b_hi, sizeof(val_b_hi));
		memcpy(&m128i_Fletcher_c_lo, val_c_lo, sizeof(val_c_lo));
		memcpy(&m128i_Fletcher_c_hi, val_c_hi, sizeof(val_c_hi));

		ui32_idx_c_ref += ui32_n;
	}

	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 4; ui32_idx_i++)
	{
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, val_b_lo[ui32_idx_i]);
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, val_b_hi[ui32_idx_i]);
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, val_c_lo[ui32_idx_i]);
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, val_c_hi[ui32_idx_i]);
	}
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher_b, Fletcher_c.ui32);
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher, Fletcher_a.ui32);

	return Fletcher.ui32;
}

/*==============================================================================================================
**									Name: smm_intel_fletcher_intermediate
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_fletcher	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_fletcher_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	// Fletcher Checksum with SIMD instructions(B and C)
	__m128i m128i_Fletcher_b_lo = _mm_setzero_si128(),
		m128i_Fletcher_b_hi = _mm_setzero_si128(),
		m128i_Fletcher_c_lo = _mm_setzero_si128(),
		m128i_Fletcher_c_hi = _mm_setzero_si128(),
		aux_128i_b_lo, aux_128i_b_hi,
		aux_128i_c_lo, aux_128i_c_hi;
	
	uint32_t val_b_lo[4] = { 0u },
		val_b_hi[4] = { 0u },
		val_c_lo[4] = { 0u },
		val_c_hi[4] = { 0u };

	/* Fletcher checksum sequential */
	ui32_to_ui16_t Fletcher_a, Fletcher_b, Fletcher_c, Fletcher;
	Fletcher_a.ui32 = 0u;
	Fletcher_b.ui32 = 0u;
	Fletcher_c.ui32 = 0u;
	Fletcher.ui32 = 0u;

	__m256i aux_256i_b, aux_256i_c;
	__m256 a256, b256, c256, result256;    // AVX

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	float f32_a_part;
	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);
			}

			// =====================================================
			//				FLETCHER Intermediate
			// =====================================================
			// Casting of the variables (B and C values)
			aux_256i_b = _mm256_castps_si256(b256);
			aux_256i_c = _mm256_castps_si256(result256);


			// Fletcher computation B
			aux_128i_b_lo = _mm256_extractf128_si256(aux_256i_b, 0);
			aux_128i_b_hi = _mm256_extractf128_si256(aux_256i_b, 1);

			m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_lo);
			m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);
			m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_hi);
			m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);


			// Fletcher computation C
			aux_128i_c_lo = _mm256_extractf128_si256(aux_256i_c, 0);
			aux_128i_c_hi = _mm256_extractf128_si256(aux_256i_c, 1);

			m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_lo);
			m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);
			m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_hi);
			m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);

			// Fletcher checksum requires a modulo operation. This algebraic operation is not implemented with
			// SIMD instructions, and therefore, it has to be implemented with sequential instructions and it will
			// produce an increment in the performance impact
			memcpy(val_b_lo, &m128i_Fletcher_b_lo, sizeof(val_b_lo));
			memcpy(val_b_hi, &m128i_Fletcher_b_hi, sizeof(val_b_hi));
			memcpy(val_c_lo, &m128i_Fletcher_c_lo, sizeof(val_c_lo));
			memcpy(val_c_hi, &m128i_Fletcher_c_hi, sizeof(val_c_hi));
			for (uint32_t ui32_idx_l = 0; ui32_idx_l < 4; ui32_idx_l++)
			{
				val_b_lo[ui32_idx_l] %= 65535;
				val_b_hi[ui32_idx_l] %= 65535;
				val_c_lo[ui32_idx_l] %= 65535;
				val_c_hi[ui32_idx_l] %= 65535;
			}
			memcpy(&m128i_Fletcher_b_lo, val_b_lo, sizeof(val_b_lo));
			memcpy(&m128i_Fletcher_b_hi, val_b_hi, sizeof(val_b_hi));
			memcpy(&m128i_Fletcher_c_lo, val_c_lo, sizeof(val_c_lo));
			memcpy(&m128i_Fletcher_c_hi, val_c_hi, sizeof(val_c_hi));

			// Evaluation of the ES (A value)
			Fletcher_a.ui32 = Fletcher32c_ui32(Fletcher_a, (uint32_t) * (uint32_t*)&f32_a_part);

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
					Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, (uint32_t) * (uint32_t *)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, (uint32_t) * (uint32_t *)&paf32_mc[ui32_idx_c + ui32_idx_j]);
				}
			}

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 4; ui32_idx_i++)
	{
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, val_b_lo[ui32_idx_i]);
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, val_b_hi[ui32_idx_i]);
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, val_c_lo[ui32_idx_i]);
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, val_c_hi[ui32_idx_i]);
	}
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher_b, Fletcher_c.ui32);
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher, Fletcher_a.ui32);

	return Fletcher.ui32;
}

/*==============================================================================================================
**									Name: smm_intel_fletcher_internal
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the internal loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 		Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 		Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 		Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_fletcher	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_fletcher_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	// Fletcher Checksum with SIMD instructions(B and C)
	__m128i m128i_Fletcher_b_lo = _mm_setzero_si128(),
		m128i_Fletcher_b_hi = _mm_setzero_si128(),
		m128i_Fletcher_c_lo = _mm_setzero_si128(),
		m128i_Fletcher_c_hi = _mm_setzero_si128(),
		aux_128i_b_lo , aux_128i_b_hi,
		aux_128i_c_lo, aux_128i_c_hi;
	// __m128i m128i_mod_255 = _mm_set1_epi32(255);

	uint32_t val_b_lo[4] = { 0u }, 
		val_b_hi[4] = { 0u },
		val_c_lo[4] = { 0u },
		val_c_hi[4] = { 0u };

	/* Fletcher checksum sequential */
	ui32_to_ui16_t Fletcher_a, Fletcher_b, Fletcher_c, Fletcher; // , Fletcher_b_lo, Fletcher_b_hi, Fletcher_c_lo, Fletcher_c_hi;
	Fletcher_a.ui32 = 0u;
	Fletcher_b.ui32 = 0u;
	Fletcher_c.ui32 = 0u;
	Fletcher.ui32 = 0u;
	/*
	Fletcher_b_lo.ui32 = 0u,
	Fletcher_b_hi.ui32 = 0u,
	Fletcher_c_lo.ui32 = 0u,
	Fletcher_c_hi.ui32 = 0u;
	*/

	ui64_to_ui32_t Fletcher_aux;
	Fletcher_aux.ui64 = 0u;

	__m256i aux_256i_b, aux_256i_c;
	__m256 a256, b256, c256, result256;    // AVX

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float f32_a_part;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);

				// Casting of the variables (B and C values)
				aux_256i_b = _mm256_castps_si256(b256);
				aux_256i_c = _mm256_castps_si256(result256);

				// =====================================================
				//				FLETCHER INTERNAL
				// =====================================================
				// Fletcher computation B
				aux_128i_b_lo = _mm256_extractf128_si256(aux_256i_b, 0);
				aux_128i_b_hi = _mm256_extractf128_si256(aux_256i_b, 1);

				m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_lo);
				m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);
				m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_hi);
				m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);


				// Fletcher computation C
				aux_128i_c_lo = _mm256_extractf128_si256(aux_256i_c, 0);
				aux_128i_c_hi = _mm256_extractf128_si256(aux_256i_c, 1);

				m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_lo);
				m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);
				m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_hi);
				m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);

				// Fletcher checksum requires a modulo operation. This algebraic operation is not implemented with
				// SIMD instructions, and therefore, it has to be implemented with sequential instructions and it will
				// produce an increment in the performance impact
				memcpy(val_b_lo, &m128i_Fletcher_b_lo, sizeof(val_b_lo));
				memcpy(val_b_hi, &m128i_Fletcher_b_hi, sizeof(val_b_hi));
				memcpy(val_c_lo, &m128i_Fletcher_c_lo, sizeof(val_c_lo));
				memcpy(val_c_hi, &m128i_Fletcher_c_hi, sizeof(val_c_hi));
				for (uint32_t ui32_idx_l = 0; ui32_idx_l < 4; ui32_idx_l++)
				{
					val_b_lo[ui32_idx_l] %= 65535;
					val_b_hi[ui32_idx_l] %= 65535;
					val_c_lo[ui32_idx_l] %= 65535;
					val_c_hi[ui32_idx_l] %= 65535;
				}
				memcpy(&m128i_Fletcher_b_lo, val_b_lo, sizeof(val_b_lo));
				memcpy(&m128i_Fletcher_b_hi, val_b_hi, sizeof(val_b_hi));
				memcpy(&m128i_Fletcher_c_lo, val_c_lo, sizeof(val_c_lo));
				memcpy(&m128i_Fletcher_c_hi, val_c_hi, sizeof(val_c_hi));
			}

			// Evaluation of the ES (A value)
			Fletcher_a.ui32 = Fletcher32c_ui32(Fletcher_a, (uint32_t) * (uint32_t*)&f32_a_part);

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
					Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, (uint32_t) * (uint32_t *)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, (uint32_t) * (uint32_t *)&paf32_mc[ui32_idx_c + ui32_idx_j]);
				}
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 4; ui32_idx_i++)
	{
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, val_b_lo[ui32_idx_i]);
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, val_b_hi[ui32_idx_i]);
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, val_c_lo[ui32_idx_i]);
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, val_c_hi[ui32_idx_i]);
	}
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher_b, Fletcher_c.ui32);
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher, Fletcher_a.ui32);

	// Fletcher computation
	/*
	m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, m128i_Fletcher_c_lo);
	m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);
	m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, m128i_Fletcher_c_hi);
	m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);

	memcpy(val_b_lo, &m128i_Fletcher_b_lo, sizeof(float32_t) * 4);
	memcpy(val_b_hi, &m128i_Fletcher_b_hi, sizeof(float32_t) * 4);
	for (uint32_t ui32_idx_l = 0; ui32_idx_l < 4; ui32_idx_l++)
	{
		Fletcher.ui32 = (uint32_t) * (uint32_t*)&val_b_lo[ui32_idx_l];
		Fletcher.ui16[0] %= 255;
		Fletcher.ui16[1] %= 255;
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b,Fletcher.ui32);

		Fletcher.ui32 = (uint32_t) * (uint32_t*)&val_b_hi[ui32_idx_l];
		Fletcher.ui16[0] %= 255;
		Fletcher.ui16[1] %= 255;
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, Fletcher.ui32);
	}
	Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, Fletcher_b.ui32);
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher_c, Fletcher_a.ui32);
	*/
	return Fletcher.ui32;
}
/*==============================================================================================================
**									Name: smm_intel_crc_external
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the external loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 		Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 		Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 		Pointer to the first position of an array of floats (B matrix direction)
**
** @return __m256  	m256_crc	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_crc_external(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	uint32_t ui32_crc_a = INITIAL_REMAINDER,
		ui32_crc_b = INITIAL_REMAINDER,
		ui32_crc_c = INITIAL_REMAINDER,
		ui32_crc = INITIAL_REMAINDER;

	__m256 a256, b256, c256, result256;    // AVX

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t val_b[8], val_c[8];

	float A_PART;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			A_PART = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(A_PART);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += A_PART * paf32_mb[ui32_idx_b + ui32_idx_j];
				}
			}
			ui32_idx_b_ref += ui32_n;
		}
		// Evaluation of the ES (A value)
		ui32_crc_a = _mm_crc32_u32(ui32_crc_a, (uint32_t) * (uint32_t*)&A_PART);

		memcpy(val_b, &b256, sizeof(float32_t) * 8);
		memcpy(val_c, &result256, sizeof(float32_t) * 8);
		for (uint32_t ui32_idx_l = 0; ui32_idx_l < 8; ui32_idx_l++)
		{
			ui32_crc_b = _mm_crc32_u32(ui32_crc_b, (uint32_t) * (uint32_t*)&val_b[ui32_idx_l]);
			ui32_crc_c = _mm_crc32_u32(ui32_crc_c, (uint32_t) * (uint32_t*)&val_c[ui32_idx_l]);
		}

		ui32_idx_c_ref += ui32_n;
	}
	ui32_crc = _mm_crc32_u32(ui32_crc_a, ui32_crc_b);
	ui32_crc = _mm_crc32_u32(ui32_crc, ui32_crc_c);
	return ui32_crc;
}

/*==============================================================================================================
**									Name: smm_intel_crc_intermediate
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha Correction factor
** @param[in] paf32_ma 	Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 	Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 	Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_crc	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_crc_intermediate(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	uint32_t ui32_crc_a = INITIAL_REMAINDER,
		ui32_crc_b = INITIAL_REMAINDER,
		ui32_crc_c = INITIAL_REMAINDER,
		ui32_crc = INITIAL_REMAINDER;

	__m256 a256, b256, c256, result256;    // AVX

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t val_b[8], val_c[8];
	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			float A_PART = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(A_PART);

			// Evaluation of the ES (A value)
			ui32_crc_a = _mm_crc32_u32(ui32_crc_a, (uint32_t) * (uint32_t*)&A_PART);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);
			}

			memcpy(val_b, &b256, sizeof(val_b));
			memcpy(val_c, &result256, sizeof(val_c));
			for (uint32_t ui32_idx_l = 0; ui32_idx_l < 8; ui32_idx_l++)
			{
				ui32_crc_b = _mm_crc32_u32(ui32_crc_b, (uint32_t) * (uint32_t*)&val_b[ui32_idx_l]);
				ui32_crc_c = _mm_crc32_u32(ui32_crc_c, (uint32_t) * (uint32_t*)&val_c[ui32_idx_l]);
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += A_PART * paf32_mb[ui32_idx_b + ui32_idx_j];
					ui32_crc_b = _mm_crc32_u32(ui32_crc_b, (uint32_t) * (uint32_t*)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					ui32_crc_c = _mm_crc32_u32(ui32_crc_c, (uint32_t) * (uint32_t*)&paf32_mc[ui32_idx_c + ui32_idx_j]);
				}
			}

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	ui32_crc = _mm_crc32_u32(ui32_crc_a, ui32_crc_b);
	ui32_crc = _mm_crc32_u32(ui32_crc, ui32_crc_c);
	return ui32_crc;
}

/*==============================================================================================================
**									Name: smm_intel_crc_internal
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the internal loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 		Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 		Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 		Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_crc	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_crc_internal(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	uint32_t ui32_crc_a = INITIAL_REMAINDER,
		ui32_crc_b = INITIAL_REMAINDER,
		ui32_crc_c = INITIAL_REMAINDER,
		ui32_crc = INITIAL_REMAINDER;

	__m256 a256, b256, c256, result256;    // AVX

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t val_b[8], val_c[8];

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			float A_PART = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(A_PART);

			// Evaluation of the ES (A value)
			ui32_crc_a = _mm_crc32_u32(ui32_crc_a, (uint32_t) * (uint32_t*)&A_PART);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);


				memcpy(val_b, &b256, sizeof(val_b));
				memcpy(val_c, &result256, sizeof(val_c));
				for (uint32_t ui32_idx_l = 0; ui32_idx_l < 8; ui32_idx_l++)
				{
					ui32_crc_b = _mm_crc32_u32(ui32_crc_b, (uint32_t) * (uint32_t*)&val_b[ui32_idx_l]);
					ui32_crc_c = _mm_crc32_u32(ui32_crc_c, (uint32_t) * (uint32_t*)&val_c[ui32_idx_l]);
				}
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += A_PART * paf32_mb[ui32_idx_b + ui32_idx_j];
					ui32_crc_b = _mm_crc32_u32(ui32_crc_b, (uint32_t) * (uint32_t*)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					ui32_crc_c = _mm_crc32_u32(ui32_crc_c, (uint32_t) * (uint32_t*)&paf32_mc[ui32_idx_c + ui32_idx_j]);
				}
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	ui32_crc = singletable_crc32c_ui32(ui32_crc_a, ui32_crc_b);
	ui32_crc = singletable_crc32c_ui32(ui32_crc, ui32_crc_c);
	return ui32_crc;
}

/* ==============================================================================================================
* 	Name: smm_gemm_nn_instrics_intel
* ============================================================================================================== */
static uint32_t smm_gemm_nn_intrincs_intel(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	__m256 a256, b256, c256, result256;    // AVX

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			float A_PART = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(A_PART);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += A_PART * paf32_mb[ui32_idx_b + ui32_idx_j];
				}
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	return 0u;
}

/*==============================================================================================================
**									Name: smm_intel_xor_flet
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the internal loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 		Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 		Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 		Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_xor	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_xor_flet(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	// ==================================
	//			XOR VARIABLES
	// ==================================
	/* XOR checksum */
	__m256i m256i_xor_b = _mm256_setzero_si256(),
		m256i_xor_c = _mm256_setzero_si256();

	/* XOR checksum */
	uint32_t ui32_xor_a = 0u,
		ui32_xor_b = 0u,
		ui32_xor_c = 0u,
		ui32_xor = 0u;

	// ==================================
	//			FLETCHER VARIABLES
	// ==================================
	/* Fletcher Checksum with SIMD instructions(B and C) */
	__m128i m128i_Fletcher_b_lo = _mm_setzero_si128(),
		m128i_Fletcher_b_hi = _mm_setzero_si128(),
		m128i_Fletcher_c_lo = _mm_setzero_si128(),
		m128i_Fletcher_c_hi = _mm_setzero_si128(),
		aux_128i_b_lo, aux_128i_b_hi,
		aux_128i_c_lo, aux_128i_c_hi;

	uint32_t val_b_lo[4] = { 0u },
		val_b_hi[4] = { 0u },
		val_c_lo[4] = { 0u },
		val_c_hi[4] = { 0u };

	/* Fletcher checksum sequential */
	ui32_to_ui16_t Fletcher_a, Fletcher_b, Fletcher_c, Fletcher, Fletcher_aux;
	Fletcher_a.ui32 = 0u;
	Fletcher_b.ui32 = 0u;
	Fletcher_c.ui32 = 0u;
	Fletcher_aux.ui32 = 0u;
	Fletcher.ui32 = 0u;

	// ==================================
	//			AUX VARIABLES
	// ==================================
	__m256 a256, b256, c256, result256;    // AVX
	float f32_a_part;

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);

				// Evaluation of the ES (B value)
				m256i_xor_b = _mm256_xor_si256(m256i_xor_b, _mm256_castps_si256(b256));

				// Evaluation of the ES (C value)
				m256i_xor_c = _mm256_xor_si256(m256i_xor_c, _mm256_castps_si256(result256));
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
					/* XOR checksum */
					ui32_xor_b ^= (uint32_t) *((uint32_t*)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					ui32_xor_c ^= (uint32_t) *((uint32_t*)&paf32_mc[ui32_idx_c + ui32_idx_j]);
				}
			}
			// Fletcher computation A
			Fletcher_a.ui32 = Fletcher32c_ui32(Fletcher_a, (uint32_t) * (uint32_t*)&f32_a_part);

			// Fletcher computation B
			aux_128i_b_lo = _mm256_extractf128_si256(m256i_xor_b, 0);
			aux_128i_b_hi = _mm256_extractf128_si256(m256i_xor_b, 1);

			m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_lo);
			m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);
			m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_hi);
			m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);


			// Fletcher computation C
			aux_128i_c_lo = _mm256_extractf128_si256(m256i_xor_c, 0);
			aux_128i_c_hi = _mm256_extractf128_si256(m256i_xor_c, 1);

			m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_lo);
			m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);
			m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_hi);
			m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);

			memcpy(val_b_lo, &m128i_Fletcher_b_lo, sizeof(val_b_lo));
			memcpy(val_b_hi, &m128i_Fletcher_b_hi, sizeof(val_b_hi));
			memcpy(val_c_lo, &m128i_Fletcher_c_lo, sizeof(val_c_lo));
			memcpy(val_c_hi, &m128i_Fletcher_c_hi, sizeof(val_c_hi));
			for (uint32_t ui32_idx_l = 0; ui32_idx_l < 4; ui32_idx_l++)
			{
				val_b_lo[ui32_idx_l] %= 65535;
				val_b_hi[ui32_idx_l] %= 65535;
				val_c_lo[ui32_idx_l] %= 65535;
				val_c_hi[ui32_idx_l] %= 65535;
			}
			memcpy(&m128i_Fletcher_b_lo, val_b_lo, sizeof(val_b_lo));
			memcpy(&m128i_Fletcher_b_hi, val_b_hi, sizeof(val_b_hi));
			memcpy(&m128i_Fletcher_c_lo, val_c_lo, sizeof(val_c_lo));
			memcpy(&m128i_Fletcher_c_hi, val_c_hi, sizeof(val_c_hi));

			if (0 != (ui32_n % 8)) {
				Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, ui32_xor_b);
				Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, ui32_xor_c);
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 4; ui32_idx_i++)
	{
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, val_b_lo[ui32_idx_i]);
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, val_b_hi[ui32_idx_i]);
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, val_c_lo[ui32_idx_i]);
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, val_c_hi[ui32_idx_i]);
	}
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher_b, Fletcher_c.ui32);
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher, Fletcher_a.ui32);

	return Fletcher.ui32;
}

/*==============================================================================================================
**									Name: smm_intel_xor_crc
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with XOR checksum in the internal loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 		Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 		Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 		Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_crc	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_xor_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	// ==================================
	//			XOR VARIABLES
	// ==================================
	/* XOR checksum */
	__m256i m256i_xor_a = _mm256_setzero_si256(),
		m256i_xor_b = _mm256_setzero_si256(),
		m256i_xor_c = _mm256_setzero_si256(),
		m256i_xor = _mm256_setzero_si256();

	/* XOR checksum */
	uint32_t ui32_xor_a = 0u,
		ui32_xor_b = 0u,
		ui32_xor_c = 0u,
		ui32_xor = 0u;

	// ==================================
	//			CRC VARIABLES
	// ==================================
	uint32_t ui32_crc_a = INITIAL_REMAINDER,
		ui32_crc_b = INITIAL_REMAINDER,
		ui32_crc_c = INITIAL_REMAINDER,
		ui32_crc = INITIAL_REMAINDER;
	uint32_t val[8], val_b[8], val_c[8];

	// ==================================
	//			AUX VARIABLES
	// ==================================
	__m256 a256, b256, c256, result256;    // AVX
	float f32_a_part;

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				//printf("\n Internal loop: idx_b:%d \tN=%d \t", ui32_idx_b, ui32_n);
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);


				// =====================================================
				//				INTERNAL XOR
				// =====================================================
				// Evaluation of the ES (B value)
				m256i_xor_b = _mm256_xor_si256(m256i_xor_b, _mm256_castps_si256(b256));

				// Evaluation of the ES (C value)
				m256i_xor_c = _mm256_xor_si256(m256i_xor_c, _mm256_castps_si256(result256));
			}

			//printf("\n Intermediate loop: idx_b:%d \tN=%d \tidx_c=%d", ui32_idx_b, ui32_n, ui32_idx_c);
			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
					/* XOR checksum */
					//printf("\n idx_c:%d \tidx_c=%d \t", ui32_idx_c + ui32_idx_j, ui32_idx_b + ui32_idx_j);
					ui32_xor_b ^= (uint32_t) *((uint32_t*)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					ui32_xor_c ^= (uint32_t) *((uint32_t*)&paf32_mc[ui32_idx_c + ui32_idx_j]);
				}
			}
			m256i_xor_a = _mm256_xor_si256(m256i_xor_a, _mm256_castps_si256(a256));
			m256i_xor = _mm256_xor_si256(m256i_xor_a, m256i_xor_b);
			m256i_xor = _mm256_xor_si256(m256i_xor  , m256i_xor_c);

			memcpy(val, &m256i_xor, sizeof(val));
			for (uint32_t ui32_idx_l = 0; ui32_idx_l < 8; ui32_idx_l++)
			{
				ui32_crc = _mm_crc32_u32(ui32_crc, (uint32_t) * (uint32_t*)&val[ui32_idx_l]);
			}

			if (0 != prev_end) {
				ui32_xor = (ui32_xor_a ^ ui32_xor_b) ^ ui32_xor_c;
				ui32_crc = _mm_crc32_u32(ui32_crc, ui32_xor);
			}


			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	return ui32_crc;
}

/*==============================================================================================================
**									Name: smm_intel_twos_flet
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with TWOS checksum in the internal loop and Fletcher checksum in the
**        intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 		Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 		Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 		Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	Fletcher.ui32	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_twos_flet(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	// ==================================
	//			ONES VARIABLES
	// ==================================
	// AVX variables
	__m256i m256i_twos_b = _mm256_setzero_si256(),
		m256i_twos_c = _mm256_setzero_si256();

	__m256i m256i_Ones = _mm256_set1_epi32(-1);
	__m256i m256i_singleOne = _mm256_set1_epi32(1);


	// C variables
	uint32_t Twos_Checksum_a = 0u,
		Twos_Checksum_b = 0u,
		Twos_Checksum_c = 0u,
		Twos_Checksum = 0u;

	// ==================================
	//			FLETCHER VARIABLES
	// ==================================
	/* Fletcher Checksum with SIMD instructions(B and C) */
	__m128i m128i_Fletcher_b_lo = _mm_setzero_si128(),
		m128i_Fletcher_b_hi = _mm_setzero_si128(),
		m128i_Fletcher_c_lo = _mm_setzero_si128(),
		m128i_Fletcher_c_hi = _mm_setzero_si128(),
		aux_128i_b_lo, aux_128i_b_hi,
		aux_128i_c_lo, aux_128i_c_hi;
	
	uint32_t val_b_lo[4] = { 0u },
		val_b_hi[4] = { 0u },
		val_c_lo[4] = { 0u },
		val_c_hi[4] = { 0u };

	/* Fletcher checksum sequential */
	ui32_to_ui16_t Fletcher_a, Fletcher_b, Fletcher_c, Fletcher, Fletcher_aux;
	Fletcher_a.ui32 = 0u;
	Fletcher_b.ui32 = 0u;
	Fletcher_c.ui32 = 0u;
	Fletcher_aux.ui32 = 0u;
	Fletcher.ui32 = 0u;


	// ==================================
	//			AUX VARIABLES
	// ==================================
	__m256 a256, b256, c256, result256;    // AVX
	float f32_a_part;

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);


				// =====================================================
				//				INTERNAL TWOS
				// =====================================================
				// Evaluation of the ES (B value)
				m256i_twos_b = _mm256_add_epi32(m256i_twos_b, _mm256_castps_si256(b256));
				m256i_twos_b = _mm256_xor_si256(m256i_twos_b, m256i_Ones);
				m256i_twos_b = _mm256_add_epi32(m256i_twos_b, m256i_singleOne);

				// Evaluation of the ES (C value)
				m256i_twos_c = _mm256_add_epi32(m256i_twos_c, _mm256_castps_si256(result256));
				m256i_twos_c = _mm256_xor_si256(m256i_twos_c, m256i_Ones);
				m256i_twos_c = _mm256_add_epi32(m256i_twos_c, m256i_singleOne);
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];

					/* Twos' checksum */
					// Evaluation of the ES (B value)
					Twos_Checksum_b += (uint32_t) *((uint32_t*)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					Twos_Checksum_b = (~Twos_Checksum_b) + 1;
					// Evaluation of the ES (C value)
					Twos_Checksum_c += (uint32_t) *((uint32_t*)&paf32_mc[ui32_idx_c + ui32_idx_j]);
					Twos_Checksum_c = (~Twos_Checksum_c) + 1;
				}
			}
			// =====================================================
			//				INTERMEDIATE FLETCHER
			// =====================================================
			// Evaluation of the ES (A value)
			Fletcher_a.ui32 = Fletcher32c_ui32(Fletcher_a, (uint32_t) * (uint32_t*)&f32_a_part);

			// Fletcher computation B
			aux_128i_b_lo = _mm256_extractf128_si256(m256i_twos_b, 0);
			aux_128i_b_hi = _mm256_extractf128_si256(m256i_twos_b, 1);

			m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_lo);
			m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);
			m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_hi);
			m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);


			// Fletcher computation C
			aux_128i_c_lo = _mm256_extractf128_si256(m256i_twos_c, 0);
			aux_128i_c_hi = _mm256_extractf128_si256(m256i_twos_c, 1);

			m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_lo);
			m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);
			m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_hi);
			m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);

			memcpy(val_b_lo, &m128i_Fletcher_b_lo, sizeof(val_b_lo));
			memcpy(val_b_hi, &m128i_Fletcher_b_hi, sizeof(val_b_hi));
			memcpy(val_c_lo, &m128i_Fletcher_c_lo, sizeof(val_c_lo));
			memcpy(val_c_hi, &m128i_Fletcher_c_hi, sizeof(val_c_hi));
			for (uint32_t ui32_idx_l = 0; ui32_idx_l < 4; ui32_idx_l++)
			{
				val_b_lo[ui32_idx_l] %= 65535;
				val_b_hi[ui32_idx_l] %= 65535;
				val_c_lo[ui32_idx_l] %= 65535;
				val_c_hi[ui32_idx_l] %= 65535;
			}
			memcpy(&m128i_Fletcher_b_lo, val_b_lo, sizeof(val_b_lo));
			memcpy(&m128i_Fletcher_b_hi, val_b_hi, sizeof(val_b_hi));
			memcpy(&m128i_Fletcher_c_lo, val_c_lo, sizeof(val_c_lo));
			memcpy(&m128i_Fletcher_c_hi, val_c_hi, sizeof(val_c_hi));

			if (0 != prev_end) {
				Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, Twos_Checksum_b);
				Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, Twos_Checksum_c);
			}
			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 4; ui32_idx_i++)
	{
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, val_b_lo[ui32_idx_i]);
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, val_b_hi[ui32_idx_i]);
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, val_c_lo[ui32_idx_i]);
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, val_c_hi[ui32_idx_i]);
	}
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher_b, Fletcher_c.ui32);
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher, Fletcher_a.ui32);

	return Fletcher.ui32;
}

/*==============================================================================================================
**									Name: smm_intel_twos_crc
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with TWO's complement checksum in the internal loop and CRC in the
**        intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 		Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 		Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 		Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_crc	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_twos_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	// ==================================
	//			TWOS VARIABLES
	// ==================================
	__m256i m256i_twos_b = _mm256_setzero_si256(),
		m256i_twos_c = _mm256_setzero_si256();

	/* Two's complement checksum */
	uint32_t Twos_Checksum_a = 0u,
		Twos_Checksum_b = 0u,
		Twos_Checksum_c = 0u,
		Twos_Checksum = 0u;

	__m256i m256i_Ones = _mm256_set1_epi32(-1);
	__m256i m256i_singleOne = _mm256_set1_epi32(1);

	// ==================================
	//			CRC VARIABLES
	// ==================================
	uint32_t ui32_crc_a = INITIAL_REMAINDER,
		ui32_crc_b = INITIAL_REMAINDER,
		ui32_crc_c = INITIAL_REMAINDER,
		ui32_crc = INITIAL_REMAINDER;
	uint32_t val_b[8], val_c[8];

	// ==================================
	//			AUX VARIABLES
	// ==================================
	__m256 a256, b256, c256, result256;    // AVX
	float f32_a_part;

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);


				// =====================================================
				//				INTERNAL TWOS
				// =====================================================
				// Evaluation of the ES (B value)
				m256i_twos_b = _mm256_add_epi32(m256i_twos_b, _mm256_castps_si256(b256));
				m256i_twos_b = _mm256_xor_si256(m256i_twos_b, m256i_Ones);
				m256i_twos_b = _mm256_add_epi32(m256i_twos_b, m256i_singleOne);

				// Evaluation of the ES (C value)
				m256i_twos_c = _mm256_add_epi32(m256i_twos_c, _mm256_castps_si256(result256));
				m256i_twos_c = _mm256_xor_si256(m256i_twos_c, m256i_Ones);
				m256i_twos_c = _mm256_add_epi32(m256i_twos_c, m256i_singleOne);
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
					/* Twos' checksum */
					Twos_Checksum_b += (uint32_t) *((uint32_t*)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					Twos_Checksum_b = (~Twos_Checksum_b) + 1;
					Twos_Checksum_c += (uint32_t) *((uint32_t*)&paf32_mc[ui32_idx_c + ui32_idx_j]);
					Twos_Checksum_c = (~Twos_Checksum_c) + 1;
				}
			}
			// =====================================================
			//				INTERMEDIATE CRC
			// =====================================================
			memcpy(val_b, &m256i_twos_b, sizeof(val_b));
			memcpy(val_c, &m256i_twos_c, sizeof(val_c));
			for (uint32_t ui32_idx_l = 0; ui32_idx_l < 8; ui32_idx_l++)
			{
				ui32_crc_b = _mm_crc32_u32(ui32_crc_b, (uint32_t) * (uint32_t*)&val_b[ui32_idx_l]);
				ui32_crc_c = _mm_crc32_u32(ui32_crc_c, (uint32_t) * (uint32_t*)&val_c[ui32_idx_l]);
			}

			if (0 != (ui32_n % 8)) {
				ui32_crc_b = _mm_crc32_u32(ui32_crc_b, Twos_Checksum_b);
				ui32_crc_c = _mm_crc32_u32(ui32_crc_c, Twos_Checksum_c);
			}
			ui32_crc_a = _mm_crc32_u32(ui32_crc_a, (uint32_t) *((uint32_t*)&f32_a_part));

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	ui32_crc = _mm_crc32_u32(ui32_crc_a, ui32_crc_b);
	ui32_crc = _mm_crc32_u32(ui32_crc, ui32_crc_c);
	return ui32_crc;
}


/*==============================================================================================================
**									Name: smm_intel_ones_crc
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with ONES checksum in the internal loop and CRC checksum in the
**        intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 		Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 		Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 		Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	ui32_crc	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_ones_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	// ==================================
	//			ONES VARIABLES
	// ==================================
	// AVX variables
	__m256i  Ones_checksum_b_hi = _mm256_setzero_si256(),
		Ones_checksum_b_lo = _mm256_setzero_si256(),
		Ones_checksum_c_hi = _mm256_setzero_si256(),
		Ones_checksum_c_lo = _mm256_setzero_si256();

	__m256i b_aux_lo, b_aux_hi,
		c_aux_lo, c_aux_hi,
		m256i_zeros = _mm256_setzero_si256();

	__m256i m256i_Ones = _mm256_set1_epi32(-1);

	// C variables
	ui64_to_ui32_t Ones_Checksum_a,
		Ones_Checksum_b,
		Ones_Checksum_c,
		Ones_Checksum;

	Ones_Checksum_a.ui64 = 0u;
	Ones_Checksum_b.ui64 = 0u;
	Ones_Checksum_c.ui64 = 0u;
	Ones_Checksum.ui64 = 0u;

	// ==================================
	//			CRC VARIABLES
	// ==================================
	uint32_t ui32_crc_a = INITIAL_REMAINDER,
		ui32_crc_b = INITIAL_REMAINDER,
		ui32_crc_c = INITIAL_REMAINDER,
		ui32_crc = INITIAL_REMAINDER;
	uint32_t val_b[8], val_c[8];

	// ==================================
	//			AUX VARIABLES
	// ==================================
	__m256 a256, b256, c256, result256;    // AVX
	float f32_a_part;

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);


				// =====================================================
				//				INTERNAL ONES COMPLEMENT
				// =====================================================
				// Evaluation of the ES (B value)
				b_aux_hi = _mm256_unpackhi_epi32(_mm256_castps_si256(b256), m256i_zeros);
				Ones_checksum_b_hi = _mm256_add_epi64(Ones_checksum_b_hi, b_aux_hi);
				Ones_checksum_b_hi = _mm256_hadd_epi32(Ones_checksum_b_hi, Ones_checksum_b_hi);
				Ones_checksum_b_hi = _mm256_xor_si256(Ones_checksum_b_hi, m256i_Ones);
				Ones_checksum_b_hi = _mm256_unpackhi_epi32(Ones_checksum_b_hi, m256i_zeros);


				b_aux_lo = _mm256_unpacklo_epi32(_mm256_castps_si256(b256), m256i_zeros);
				Ones_checksum_b_lo = _mm256_add_epi64(Ones_checksum_b_lo, b_aux_lo);
				Ones_checksum_b_lo = _mm256_hadd_epi32(Ones_checksum_b_lo, Ones_checksum_b_lo);
				Ones_checksum_b_lo = _mm256_xor_si256(Ones_checksum_b_lo, m256i_Ones);
				Ones_checksum_b_lo = _mm256_unpackhi_epi32(Ones_checksum_b_lo, m256i_zeros);


				// Evaluation of the ES (C value)
				c_aux_hi = _mm256_unpackhi_epi32(_mm256_castps_si256(result256), m256i_zeros);
				Ones_checksum_c_hi = _mm256_add_epi64(Ones_checksum_c_hi, c_aux_hi);
				Ones_checksum_c_hi = _mm256_hadd_epi32(Ones_checksum_c_hi, Ones_checksum_c_hi);
				Ones_checksum_c_hi = _mm256_xor_si256(Ones_checksum_c_hi, m256i_Ones);
				Ones_checksum_c_hi = _mm256_unpackhi_epi32(Ones_checksum_c_hi, m256i_zeros);


				c_aux_lo = _mm256_unpacklo_epi32(_mm256_castps_si256(result256), m256i_zeros);
				Ones_checksum_c_lo = _mm256_add_epi64(Ones_checksum_c_lo, c_aux_lo);
				Ones_checksum_c_lo = _mm256_hadd_epi32(Ones_checksum_c_lo, Ones_checksum_c_lo);
				Ones_checksum_c_lo = _mm256_xor_si256(Ones_checksum_c_lo, m256i_Ones);
				Ones_checksum_c_lo = _mm256_unpackhi_epi32(Ones_checksum_c_lo, m256i_zeros);
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
					/* One's complement checksum */
					Ones_Checksum_b.ui64 += (uint64_t) * ((uint32_t*)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					Ones_Checksum_b.ui32[0] += Ones_Checksum_b.ui32[1];
					Ones_Checksum_b.ui32[0] = ~Ones_Checksum_b.ui32[0];

					Ones_Checksum_c.ui64 += (uint64_t) * ((uint32_t*)&paf32_mc[ui32_idx_c + ui32_idx_j]);
					Ones_Checksum_c.ui32[0] += Ones_Checksum_c.ui32[1];
					Ones_Checksum_c.ui32[0] = ~Ones_Checksum_c.ui32[0];
				}
			}
			// =====================================================
			//				INTERMEDIATE CRC
			// =====================================================
			// Evaluation of the ES (C value)
			Ones_checksum_c_hi = _mm256_add_epi64(Ones_checksum_c_hi, Ones_checksum_c_lo);
			Ones_checksum_c_hi = _mm256_hadd_epi32(Ones_checksum_c_hi, Ones_checksum_c_hi);
			Ones_checksum_c_hi = _mm256_xor_si256(Ones_checksum_c_hi, m256i_Ones);
			Ones_checksum_c_hi = _mm256_unpackhi_epi32(Ones_checksum_c_hi, m256i_zeros);

			Ones_checksum_c_lo = _mm256_add_epi64(Ones_checksum_b_hi, Ones_checksum_b_lo);
			Ones_checksum_c_lo = _mm256_hadd_epi32(Ones_checksum_c_lo, Ones_checksum_c_lo);
			Ones_checksum_c_lo = _mm256_xor_si256(Ones_checksum_c_lo, m256i_Ones);
			Ones_checksum_c_lo = _mm256_unpackhi_epi32(Ones_checksum_c_lo, m256i_zeros);


			memcpy(val_b, &Ones_checksum_c_hi, sizeof(val_b));
			memcpy(val_c, &Ones_checksum_c_lo, sizeof(val_c));
			for (uint32_t ui32_idx_l = 0; ui32_idx_l < 8; ui32_idx_l++)
			{
				ui32_crc_b = _mm_crc32_u32(ui32_crc_b, (uint32_t) * (uint32_t*)&val_b[ui32_idx_l]);
				ui32_crc_c = _mm_crc32_u32(ui32_crc_c, (uint32_t) * (uint32_t*)&val_c[ui32_idx_l]);
			}

			if (0 != prev_end) {
				ui32_crc_b = _mm_crc32_u32(ui32_crc_b, Ones_Checksum_b.ui32[0]);
				ui32_crc_c = _mm_crc32_u32(ui32_crc_c, Ones_Checksum_c.ui32[0]);
			}
			ui32_crc_a = _mm_crc32_u32(ui32_crc_a, (uint32_t) *((uint32_t*)&f32_a_part));

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	ui32_crc = _mm_crc32_u32(ui32_crc_a, ui32_crc_b);
	ui32_crc = _mm_crc32_u32(ui32_crc, ui32_crc_c);
	return ui32_crc;
}

/*==============================================================================================================
**									Name: smm_intel_ones_flet
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with ONES checksum in the internal loop and Fletcher checksum in the
**        intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 		Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 		Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 		Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	Fletcher.ui32	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_ones_flet(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	// ==================================
	//			ONES VARIABLES
	// ==================================
	// AVX variables
	__m256i  Ones_checksum_b_hi = _mm256_setzero_si256(),
		Ones_checksum_b_lo = _mm256_setzero_si256(),
		Ones_checksum_c_hi = _mm256_setzero_si256(),
		Ones_checksum_c_lo = _mm256_setzero_si256();

	__m256i b_aux_lo, b_aux_hi,
		c_aux_lo, c_aux_hi,
		m256i_zeros = _mm256_setzero_si256();

	__m256i m256i_Ones = _mm256_set1_epi32(-1);

	// C variables
	ui64_to_ui32_t Ones_Checksum_a,
		Ones_Checksum_b,
		Ones_Checksum_c,
		Ones_Checksum;

	Ones_Checksum_a.ui64 = 0u;
	Ones_Checksum_b.ui64 = 0u;
	Ones_Checksum_c.ui64 = 0u;
	Ones_Checksum.ui64 = 0u;

	// ==================================
	//			FLETCHER VARIABLES
	// ==================================
	/* Fletcher Checksum with SIMD instructions(B and C) */
	__m128i m128i_Fletcher_b_lo = _mm_setzero_si128(),
		m128i_Fletcher_b_hi = _mm_setzero_si128(),
		m128i_Fletcher_c_lo = _mm_setzero_si128(),
		m128i_Fletcher_c_hi = _mm_setzero_si128(),
		aux_128i_b_lo, aux_128i_b_hi,
		aux_128i_c_lo, aux_128i_c_hi;
	
	uint32_t val_b_lo[4] = { 0u },
		val_b_hi[4] = { 0u },
		val_c_lo[4] = { 0u },
		val_c_hi[4] = { 0u };

	/* Fletcher checksum sequential */
	ui32_to_ui16_t Fletcher_a, Fletcher_b, Fletcher_c, Fletcher, Fletcher_aux;
	Fletcher_a.ui32 = 0u;
	Fletcher_b.ui32 = 0u;
	Fletcher_c.ui32 = 0u;
	Fletcher_aux.ui32 = 0u;
	Fletcher.ui32 = 0u;


	// ==================================
	//			AUX VARIABLES
	// ==================================
	__m256 a256, b256, c256, result256;    // AVX
	float f32_a_part;

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;


	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);


				// =====================================================
				//				INTERNAL ONES COMPLEMENT
				// =====================================================
				// Evaluation of the ES (B value)
				b_aux_hi = _mm256_unpackhi_epi32(_mm256_castps_si256(b256), m256i_zeros);
				Ones_checksum_b_hi = _mm256_add_epi64(Ones_checksum_b_hi, b_aux_hi);
				Ones_checksum_b_hi = _mm256_hadd_epi32(Ones_checksum_b_hi, Ones_checksum_b_hi);
				Ones_checksum_b_hi = _mm256_xor_si256(Ones_checksum_b_hi, m256i_Ones);
				Ones_checksum_b_hi = _mm256_unpackhi_epi32(Ones_checksum_b_hi, m256i_zeros);


				b_aux_lo = _mm256_unpacklo_epi32(_mm256_castps_si256(b256), m256i_zeros);
				Ones_checksum_b_lo = _mm256_add_epi64(Ones_checksum_b_lo, b_aux_lo);
				Ones_checksum_b_lo = _mm256_hadd_epi32(Ones_checksum_b_lo, Ones_checksum_b_lo);
				Ones_checksum_b_lo = _mm256_xor_si256(Ones_checksum_b_lo, m256i_Ones);
				Ones_checksum_b_lo = _mm256_unpackhi_epi32(Ones_checksum_b_lo, m256i_zeros);


				// Evaluation of the ES (C value)
				c_aux_hi = _mm256_unpackhi_epi32(_mm256_castps_si256(result256), m256i_zeros);
				Ones_checksum_c_hi = _mm256_add_epi64(Ones_checksum_c_hi, c_aux_hi);
				Ones_checksum_c_hi = _mm256_hadd_epi32(Ones_checksum_c_hi, Ones_checksum_c_hi);
				Ones_checksum_c_hi = _mm256_xor_si256(Ones_checksum_c_hi, m256i_Ones);
				Ones_checksum_c_hi = _mm256_unpackhi_epi32(Ones_checksum_c_hi, m256i_zeros);


				c_aux_lo = _mm256_unpacklo_epi32(_mm256_castps_si256(result256), m256i_zeros);
				Ones_checksum_c_lo = _mm256_add_epi64(Ones_checksum_c_lo, c_aux_lo);
				Ones_checksum_c_lo = _mm256_hadd_epi32(Ones_checksum_c_lo, Ones_checksum_c_lo);
				Ones_checksum_c_lo = _mm256_xor_si256(Ones_checksum_c_lo, m256i_Ones);
				Ones_checksum_c_lo = _mm256_unpackhi_epi32(Ones_checksum_c_lo, m256i_zeros);
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
					/* One's complement checksum */
					Ones_Checksum_b.ui64 += (uint64_t) * ((uint32_t*)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					Ones_Checksum_b.ui32[0] += Ones_Checksum_b.ui32[1];
					Ones_Checksum_b.ui32[0] = ~Ones_Checksum_b.ui32[0];

					Ones_Checksum_c.ui64 += (uint64_t) * ((uint32_t*)&paf32_mc[ui32_idx_c + ui32_idx_j]);
					Ones_Checksum_c.ui32[0] += Ones_Checksum_c.ui32[1];
					Ones_Checksum_c.ui32[0] = ~Ones_Checksum_c.ui32[0];
				}
			}
			// =====================================================
			//				INTERMEDIATE FLETCHER
			// =====================================================
			// Evaluation of the ES (C value)
			Ones_checksum_c_hi = _mm256_add_epi64(Ones_checksum_c_hi, Ones_checksum_c_lo);
			Ones_checksum_c_hi = _mm256_hadd_epi32(Ones_checksum_c_hi, Ones_checksum_c_hi);
			Ones_checksum_c_hi = _mm256_xor_si256(Ones_checksum_c_hi, m256i_Ones);
			Ones_checksum_c_hi = _mm256_unpackhi_epi32(Ones_checksum_c_hi, m256i_zeros);

			Ones_checksum_c_lo = _mm256_add_epi64(Ones_checksum_b_hi, Ones_checksum_b_lo);
			Ones_checksum_c_lo = _mm256_hadd_epi32(Ones_checksum_c_lo, Ones_checksum_c_lo);
			Ones_checksum_c_lo = _mm256_xor_si256(Ones_checksum_c_lo, m256i_Ones);
			Ones_checksum_c_lo = _mm256_unpackhi_epi32(Ones_checksum_c_lo, m256i_zeros);

			// Evaluation of the ES (A value)
			Fletcher_a.ui32 = Fletcher32c_ui32(Fletcher_a, (uint32_t) * (uint32_t*)&f32_a_part);

			// Fletcher computation B
			aux_128i_b_lo = _mm256_extractf128_si256(Ones_checksum_c_hi, 0);
			aux_128i_b_hi = _mm256_extractf128_si256(Ones_checksum_c_hi, 1);

			m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_lo);
			m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);
			m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_hi);
			m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);


			// Fletcher computation C
			aux_128i_c_lo = _mm256_extractf128_si256(Ones_checksum_c_lo, 0);
			aux_128i_c_hi = _mm256_extractf128_si256(Ones_checksum_c_lo, 1);

			m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_lo);
			m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);
			m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_hi);
			m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);

			memcpy(val_b_lo, &m128i_Fletcher_b_lo, sizeof(val_b_lo));
			memcpy(val_b_hi, &m128i_Fletcher_b_hi, sizeof(val_b_hi));
			memcpy(val_c_lo, &m128i_Fletcher_c_lo, sizeof(val_c_lo));
			memcpy(val_c_hi, &m128i_Fletcher_c_hi, sizeof(val_c_hi));
			for (uint32_t ui32_idx_l = 0; ui32_idx_l < 4; ui32_idx_l++)
			{
				val_b_lo[ui32_idx_l] %= 65535;
				val_b_hi[ui32_idx_l] %= 65535;
				val_c_lo[ui32_idx_l] %= 65535;
				val_c_hi[ui32_idx_l] %= 65535;
			}
			memcpy(&m128i_Fletcher_b_lo, val_b_lo, sizeof(val_b_lo));
			memcpy(&m128i_Fletcher_b_hi, val_b_hi, sizeof(val_b_hi));
			memcpy(&m128i_Fletcher_c_lo, val_c_lo, sizeof(val_c_lo));
			memcpy(&m128i_Fletcher_c_hi, val_c_hi, sizeof(val_c_hi));

			if (0 != (ui32_n % 8)) {
				Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, Ones_Checksum_b.ui32[0]);
				Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, Ones_Checksum_c.ui32[0]);
			}
			// Evaluation of the ES (A value)
			Fletcher_a.ui32 = Fletcher32c_ui32(Fletcher_a, (uint32_t) * (uint32_t*)&f32_a_part);

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	for (uint32_t ui32_idx_i = 0; ui32_idx_i < 4; ui32_idx_i++)
	{
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, val_b_lo[ui32_idx_i]);
		Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, val_b_hi[ui32_idx_i]);
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, val_c_lo[ui32_idx_i]);
		Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, val_c_hi[ui32_idx_i]);
	}
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher_b, Fletcher_c.ui32);
	Fletcher.ui32 = Fletcher32c_ui32(Fletcher, Fletcher_a.ui32);
	return Fletcher.ui32;
}


/*==============================================================================================================
**									Name: smm_intel_flet_crc
==============================================================================================================*/
/*!
** @brief Matrix-matrix multiplication (MMM) with ONES checksum in the internal loop and Fletcher checksum in the
**        intermediate loop
**
** @param[in] ui32_m 		Number of matrix A rows 								[0…ui32_m]
** @param[in] ui32_n 		Number of matrix B columns 								[0…ui32_n]
** @param[in] ui32_k 		Number of matrix A columns / Number of matrix B rows 	[0…ui32_k]
** @param[in] f32_alpha		Correction factor
** @param[in] paf32_ma 		Pointer to the first position of an array of floats (A matrix direction)
** @param[in] paf32_mb 		Pointer to the first position of an array of floats (B matrix direction)
** @param[in] paf32_mc 		Pointer to the first position of an array of floats (B matrix direction)
**
** @return uint32_t  	Fletcher.ui32	Return the Execution signature of the MMM
==============================================================================================================*/
static uint32_t smm_intel_flet_crc(uint32_t ui32_m, uint32_t ui32_n, uint32_t ui32_k, float32_t f32_alpha, const float32_t * const paf32_ma, const float32_t * const paf32_mb, float32_t * const paf32_mc)
{
	// ==================================
	//			FLETCHER VARIABLES
	// ==================================
	// Fletcher Checksum with SIMD instructions(B and C)
	__m128i m128i_Fletcher_b_lo = _mm_setzero_si128(),
		m128i_Fletcher_b_hi = _mm_setzero_si128(),
		m128i_Fletcher_c_lo = _mm_setzero_si128(),
		m128i_Fletcher_c_hi = _mm_setzero_si128(),
		aux_128i_b_lo, aux_128i_b_hi,
		aux_128i_c_lo, aux_128i_c_hi;
	// __m128i m128i_mod_255 = _mm_set1_epi32(255);

	/* Fletcher checksum sequential */
	ui32_to_ui16_t Fletcher_a, Fletcher_b, Fletcher_c, Fletcher;
	Fletcher_a.ui32 = 0u;
	Fletcher_b.ui32 = 0u;
	Fletcher_c.ui32 = 0u;
	Fletcher.ui32 = 0u;

	uint32_t val_b_lo[4], val_b_hi[4], val_c_lo[4], val_c_hi[4];

	// ==================================
	//			CRC VARIABLES
	// ==================================
	uint32_t ui32_crc_a = INITIAL_REMAINDER,
		ui32_crc_b = INITIAL_REMAINDER,
		ui32_crc_c = INITIAL_REMAINDER,
		ui32_crc = INITIAL_REMAINDER;

	__m256i aux_256i_b, aux_256i_c;
	__m256 a256, b256, c256, result256;    // AVX

	uint32_t ui32_idx_i = 0u,
		ui32_idx_j = 0u,
		ui32_idx_k = 0u,
		ui32_idx_a = 0u,
		ui32_idx_b = 0u,
		ui32_idx_c = 0u,
		ui32_idx_b_ref = 0u,
		ui32_idx_c_ref = 0u;

	float f32_a_part;
	uint32_t prev_end;
	prev_end = (ui32_n % 8);

	for (ui32_idx_i = 0; ui32_idx_i < ui32_m; ++ui32_idx_i)
	{
		ui32_idx_b_ref = 0u;
		for (ui32_idx_k = 0; ui32_idx_k < ui32_k; ++ui32_idx_k, ui32_idx_a++)
		{
			f32_a_part = f32_alpha * paf32_ma[ui32_idx_a];
			a256 = _mm256_set1_ps(f32_a_part);

			for (ui32_idx_j = 0u, ui32_idx_b = ui32_idx_b_ref, ui32_idx_c = ui32_idx_c_ref; ui32_idx_j <= (ui32_n - 8); ui32_idx_j += 8, ui32_idx_b += 8, ui32_idx_c += 8)
			{
				b256 = _mm256_loadu_ps(&paf32_mb[ui32_idx_b]);
				c256 = _mm256_loadu_ps(&paf32_mc[ui32_idx_c]);
				// FMA - Intel Haswell (2013), AMD Piledriver (2012)
				//result256 = _mm256_fmadd_ps(a256, b256, c256);
				result256 = _mm256_mul_ps(a256, b256);

				result256 = _mm256_add_ps(result256, c256);
				_mm256_storeu_ps(&paf32_mc[ui32_idx_c], result256);

				// Casting of the variables (B and C values)
				aux_256i_b = _mm256_castps_si256(b256);
				aux_256i_c = _mm256_castps_si256(result256);


				// =====================================================
				//				FLETCHER INTERNAL
				// =====================================================
				// Fletcher computation B
				aux_128i_b_lo = _mm256_extractf128_si256(aux_256i_b, 0);
				aux_128i_b_hi = _mm256_extractf128_si256(aux_256i_b, 1);

				m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_lo);
				m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);
				m128i_Fletcher_b_lo = _mm_add_epi32(m128i_Fletcher_b_lo, aux_128i_b_hi);
				m128i_Fletcher_b_hi = _mm_add_epi32(m128i_Fletcher_b_hi, m128i_Fletcher_b_lo);


				// Fletcher computation C
				aux_128i_c_lo = _mm256_extractf128_si256(aux_256i_c, 0);
				aux_128i_c_hi = _mm256_extractf128_si256(aux_256i_c, 1);

				m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_lo);
				m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);
				m128i_Fletcher_c_lo = _mm_add_epi32(m128i_Fletcher_c_lo, aux_128i_c_hi);
				m128i_Fletcher_c_hi = _mm_add_epi32(m128i_Fletcher_c_hi, m128i_Fletcher_c_lo);

				// Fletcher checksum requires a modulo operation. This algebraic operation is not implemented with
				// SIMD instructions, and therefore, it has to be implemented with sequential instructions and it will
				// produce an increment in the performance impact
				memcpy(val_b_lo, &m128i_Fletcher_b_lo, sizeof(val_b_lo));
				memcpy(val_b_hi, &m128i_Fletcher_b_hi, sizeof(val_b_hi));
				memcpy(val_c_lo, &m128i_Fletcher_c_lo, sizeof(val_c_lo));
				memcpy(val_c_hi, &m128i_Fletcher_c_hi, sizeof(val_c_hi));
				for (uint32_t ui32_idx_l = 0; ui32_idx_l < 4; ui32_idx_l++)
				{
					val_b_lo[ui32_idx_l] %= 65535;
					val_b_hi[ui32_idx_l] %= 65535;
					val_c_lo[ui32_idx_l] %= 65535;
					val_c_hi[ui32_idx_l] %= 65535;
				}
				memcpy(&m128i_Fletcher_b_lo, val_b_lo, sizeof(val_b_lo));
				memcpy(&m128i_Fletcher_b_hi, val_b_hi, sizeof(val_b_hi));
				memcpy(&m128i_Fletcher_c_lo, val_c_lo, sizeof(val_c_lo));
				memcpy(&m128i_Fletcher_c_hi, val_c_hi, sizeof(val_c_hi));
			}

			if (0 != prev_end) {
				for (ui32_idx_j = 0; ui32_idx_j < prev_end; ++ui32_idx_j) {
					paf32_mc[ui32_idx_c + ui32_idx_j] += f32_a_part * paf32_mb[ui32_idx_b + ui32_idx_j];
					Fletcher_b.ui32 = Fletcher32c_ui32(Fletcher_b, (uint32_t) * (uint32_t *)&paf32_mb[ui32_idx_b + ui32_idx_j]);
					Fletcher_c.ui32 = Fletcher32c_ui32(Fletcher_c, (uint32_t) * (uint32_t *)&paf32_mc[ui32_idx_c + ui32_idx_j]);
				}
			}
			// =====================================================
			//				INTERMEDIATE CRC
			// =====================================================
			// CRC computation C
			ui32_crc_a = _mm_crc32_u32(ui32_crc_a, (uint32_t) *((uint32_t*)&f32_a_part));

			for (uint32_t ui32_idx_l = 0; ui32_idx_l < 4; ui32_idx_l++)
			{
				ui32_crc_b = _mm_crc32_u32(ui32_crc_b, (uint32_t) * (uint32_t*)&val_b_lo[ui32_idx_l]);
				ui32_crc_b = _mm_crc32_u32(ui32_crc_b, (uint32_t) * (uint32_t*)&val_b_hi[ui32_idx_l]);
				ui32_crc_c = _mm_crc32_u32(ui32_crc_c, (uint32_t) * (uint32_t*)&val_c_lo[ui32_idx_l]);
				ui32_crc_c = _mm_crc32_u32(ui32_crc_c, (uint32_t) * (uint32_t*)&val_c_hi[ui32_idx_l]);
			}

			if (0 != prev_end) {
				ui32_crc_b = _mm_crc32_u32(ui32_crc_b, Fletcher_b.ui32);
				ui32_crc_c = _mm_crc32_u32(ui32_crc_c, Fletcher_c.ui32);
			}

			ui32_idx_b_ref += ui32_n;
		}
		ui32_idx_c_ref += ui32_n;
	}
	ui32_crc = _mm_crc32_u32(ui32_crc_a, ui32_crc_b);
	ui32_crc = _mm_crc32_u32(ui32_crc, ui32_crc_c);
	return ui32_crc;
}
