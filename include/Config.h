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

#define VERBOSE_LEVEL_0

#define VERBOSE_LEVEL_1

//#define VERBOSE_LEVEL_2

//#define VERBOSE_LEVEL_3

//#define VERBOSE_LEVEL_4

#define FUNCTIONS_MKB_ORDER_0

//#define FUNCTIONS_MKB_ORDER_2

#define IMG_VOL_BOUNDARY_NO_CHECK

#define MATRIX_BOUNDARY_NO_CHECK

#define NOISE_ZERO_MEAN

#define RECONSTRUCTOR_ASSERT_CHECK

//#define RECONSTRUCTOR_MKB_KERNEL

#define RECONSTRUCTOR_TRILINEAR_KERNEL

#define RECONSTRUCTOR_ADD_T_DURING_INSERT

//#define RECONSTRUCTOR_CHECK_C_AVERAGE

#define RECONSTRUCTOR_CHECK_C_MAX

#define RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

//#define RECONSTRUCTOR_SYMMETRIZE_DURING_INSERT

#define RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT

#define RECONSTRUCTOR_WIENER_FILTER_FSC

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
#define RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
#endif

//#define RECONSTRUCTOR_WIENER_FITLER_CONST

//#define RECONSTRUCTOR_REMOVE_CORNER

#ifdef RECONSTRUCTOR_REMOVE_CORNER
#define RECONSTRUCTOR_REMOVE_CORNER_MASK_ZERO
#endif

//#define RECONSTRUCTOR_REMOVE_NEG

//#define RECONSTRUCTOR_NORMALISE_T_F

//#define RECONSTRUCTOR_ALWAYS_JOIN_HALF

//#define PROJECTOR_REMOVE_NEG

//#define MODEL_AVERAGE_TWO_HEMISPHERE

#ifndef MODEL_AVERAGE_TWO_HEMISPHERE
//#define MODEL_RESOLUTION_BASE_AVERAGE
#endif

//#define MODEL_SWAP_HEMISPHERE

//#define MODEL_ALWAYS_MAX_RU

//#define OPTIMISER_LOW_PASS_REF

#define OPTIMISER_NORM_CORRECTION

#define OPTIMISER_REFRESH_SIGMA

//#define OPTIMISER_EXPECTATION_REMOVE_TAIL

#ifdef NOISE_ZERO_MEAN
//#define OPTIMISER_ADJUST_2D_IMAGE_NOISE_ZERO_MEAN
#endif

#define OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION

#define OPTIMISER_RECONSTRUCT_WITH_UNMASK_IMAGE

//#define OPTIMISER_DYNAMIC_NUM_SAMPLE

//#define OPTIMISER_SIGMA_MASK
//#define OPTIMISER_SCALE_MASK
#define OPTIMISER_NORM_MASK

#define OPTIMISER_SOLVENT_FLATTEN

#ifdef OPTIMISER_SOLVENT_FLATTEN
#define OPTIMISER_SOLVENT_FLATTEN_STAT_REMOVE_BG
//#define OPTIMISER_SOLVENT_FLATTEN_SUBTRACT_BG
//#define OPTIMISER_SOLVENT_FLATTEN_REMOVE_NEG
#define OPTIMISER_SOLVENT_FLATTEN_MASK_ZERO
#endif

//#define OPTIMISER_MASK_IMG

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
//#define PARTICLE_CAL_VARI_TRANS_ZERO_MEAN
#endif

//#define OPTIMISER_FINAL_PERTURBATION

//#define OPTIMISER_CTF_WRAP

//#define OPTIMISER_HARSH_SEARCH

#define OPTIMISER_SIGMA_WHOLE_FREQUENCY

//#define OPTIMISER_COMPRESS_CRITERIA

#define PARTICLE_TRANS_INIT_GAUSSIAN

//#define PARTILCE_TRANS_INIT_FLAT

#endif // CONFIG_H
