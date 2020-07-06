/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description: some macros
 *
 * Manual:
 * ****************************************************************************/

#ifndef CONFIG_H
#define CONFIG_H

#include "THUNDERConfig.h"

#ifdef GPU_VERSION

#define GPU_EXPECTATION
#define GPU_INSERT
#define GPU_RECONSTRUCT
#define GPU_REMASK

#define GPU_ERROR_CHECK

#endif

#define VERBOSE_LEVEL_0

// #define VERBOSE_LEVEL_1

//#define VERBOSE_LEVEL_2

//#define VERBOSE_LEVEL_3

//#define VERBOSE_LEVEL_4

#define FUNCTIONS_MKB_ORDER_0

//#define FUNCTIONS_MKB_ORDER_2

#define IMG_VOL_BOUNDARY_NO_CHECK

#define IMG_VOL_BOX_UNFOLD

//#define INTERP_CELL_UNFOLD

#define MATRIX_BOUNDARY_NO_CHECK

#define NAN_NO_CHECK

#define NOISE_ZERO_MEAN

#define DATABASE_SHUFFLE

#define PARTICLE_TRANS_INIT_GAUSSIAN

//#define PARTICLE_TRANS_INIT_FLAT

#define PARTICLE_DEFOCUS_INIT_GAUSSIAN

//#define PARTICLE_DEFOCUS_INIT_FLAT

#define PARTICLE_PRIOR_ONE

#define PARTICLE_RECENTRE

#ifdef PARTICLE_RECENTRE
#define PARTICLE_RECENTRE_TRANSQ
#endif

//#define PARTICLE_ROTATION_KAPPA

//#define PARTICLE_TRANSLATION_S

//#define PARTICLE_CAL_VARI_DEFOCUS_ZERO_MEAN

#define PARTICLE_ROT_MEAN_USING_STAT_CAL_VARI

#define PARTICLE_ROT_MEAN_USING_STAT_PERTURB

//#define PARTICLE_RHO

#define PARTICLE_BALANCE_WEIGHT_R

#define PARTICLE_BALANCE_WEIGHT_T

#define PARTICLE_BALANCE_WEIGHT_D

#define PARTICLE_PEAK_FACTOR_C

//#define RECONSTRUCTOR_KERNEL_PADDING

//#define RECONSTRUCTOR_ASSERT_CHECK

//#define RECONSTRUCTOR_MKB_KERNEL

#define RECONSTRUCTOR_TRILINEAR_KERNEL

#define RECONSTRUCTOR_ADD_T_DURING_INSERT

#define RECONSTRUCTOR_CHECK_C_AVERAGE

// #define RECONSTRUCTOR_CHECK_C_MAX

#define RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#define RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT

#define RECONSTRUCTOR_WIENER_FILTER_FSC

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
//#define RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
#endif

//#define RECONSTRUCTOR_REMOVE_NEG

//#define RECONSTRUCTOR_ALWAYS_JOIN_HALF

//#define RECONSTRUCTOR_LOW_PASS

//#define PROJECTOR_REMOVE_NEG

#define PROJECTOR_CORRECT_CONVOLUTION_KERNEL

//#define MODEL_AVERAGE_TWO_HEMISPHERE

#ifndef MODEL_AVERAGE_TWO_HEMISPHERE
#define MODEL_RESOLUTION_BASE_AVERAGE
#endif

//#define MODEL_SWAP_HEMISPHERE

//#define MODEL_ALWAYS_MAX_RU

//#define MODEL_ALWAYS_MAX_RU_EXCEPT_GLOBAL

//#define MODEL_DETERMINE_INCREASE_R_R_CHANGE

#define MODEL_DETERMINE_INCREASE_R_T_VARI

//#define MODEL_DETERMINE_INCREASE_FSC

#define OPTIMISER_CTF_ON_THE_FLY

#define OPTIMISER_LOG_MEM_USAGE

#define OPTIMISER_PARTICLE_FILTER

#define OPTIMISER_NORM_CORRECTION

#define OPTIMISER_REFRESH_SIGMA

//#define OPTIMISER_CORRECT_SCALE

//#define OPTIMISER_EXPECTATION_REMOVE_TAIL

//#define OPTIMISER_EXPECTATION_REMOVE_AUXILIARY_CLASS

#ifdef NOISE_ZERO_MEAN
//#define OPTIMISER_ADJUST_2D_IMAGE_NOISE_ZERO_MEAN
#endif

#define OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION

//#define OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE

//#define OPTIMISER_SIGMA_MASK
//#define OPTIMISER_SCALE_MASK
#define OPTIMISER_NORM_MASK

#define OPTIMISER_SOLVENT_FLATTEN

#ifdef OPTIMISER_SOLVENT_FLATTEN
//#define OPTIMISER_SOLVENT_FLATTEN_LOW_PASS
//#define OPTIMISER_SOLVENT_FLATTEN_STAT_REMOVE_BG
//#define OPTIMISER_SOLVENT_FLATTEN_SUBTRACT_BG
//#define OPTIMISER_SOLVENT_FLATTEN_REMOVE_NEG
//#define OPTIMISER_SOLVENT_FLATTEN_LOW_PASS_MASK
#define OPTIMISER_SOLVENT_FLATTEN_MASK_ZERO
#endif

#define OPTIMISER_MASK_IMG

#ifdef OPTIMISER_MASK_IMG
//#define OPTIMISER_SIGMA_CORE
#endif

#define OPTIMISER_INIT_IMG_NORMALISE_OUT_MASK_REGION

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
//#define PARTICLE_CAL_VARI_TRANS_ZERO_MEAN
#endif

#define OPTIMISER_SIGMA_WHOLE_FREQUENCY

#define OPTIMISER_SIGMA_RANK1ST

//#define OPTIMISER_SIGMA_GRADING

#define OPTIMISER_RECONSTRUCT_FREE_IMG_STACK_TO_SAVE_MEM

#define OPTIMISER_RECONSTRUCT_JOIN_HALF

#define OPTIMISER_2D_GRID_CORR

#define OPTIMISER_3D_GRID_CORR

#define OPTIMISER_2D_SAVE_JOIN_MAP

#define OPTIMISER_3D_SAVE_JOIN_MAP

#define OPTIMISER_PEAK_FACTOR_C

#define OPTIMISER_PEAK_FACTOR_R

//#define OPTIMISER_PEAK_FACTOR_T

//#define OPTIMISER_PEAK_FACTOR_D

#define OPTIMISER_COMPRESS_CRITERIA

#define OPTIMISER_SCAN_SET_MIN_STD_WITH_PERTURB

//#define OPTIMISER_GLOBAL_PERTURB_LARGE

//#define OPTIMISER_REFRESH_VARIANCE_BEST_CLASS

//#define OPTIMISER_SAVE_LOW_PASS_REFERENCE

//#define OPTIMISER_SAVE_IMAGES

//#define OPTIMISER_SAVE_PARTICLES

//#define OPTIMISER_SAVE_BEST_PROJECTIONS

//#define OPTIMISER_SAVE_SIGMA

#define OPTIMISER_SAVE_FSC

//#define OPTIMISER_REFRESH_SCALE_SPECTRUM

//#define OPTIMISER_REFRESH_SCALE_ZERO_FREQ_NO_COORD

#define OPTIMISER_REFRESH_SCALE_ABSOLUTE

#define OPTIMISER_BALANCE_CLASS

#endif // CONFIG_H
