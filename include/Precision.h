/** @file
 *  @author Huabin Ruan
 *  @version 1.4.11.180913
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Huabin Ruan | 2018/09/13 | 1.4.11.080913 | Add header for file and functions
 *  Mingxu Hu   | 2018/09/28 | 1.4.11.080928 | Add mathmatical background of some scientific functions
 *  Mingxu Hu   | 2018/12/22 | 1.4.11.081222 | Add some emendation in the documentation
 *
 *  @brief Precision.h encapsulates the header files/MACRO/data structures/functions of the single-precision version and the double-precision version.
 *
 *  Random Number Distributions
 *  ===========================
 *
 * Continuous random number distributions are defined by a probability density function, @f$p(x)@f$, such that the probability of @f$x@f$ occurring in the infinitesimal range @f$x@f$ to @f$x + \mathrm{d}x@f$ is @f$p\mathrm{d}x@f$.
 *
 * The cumulative distribution function for the lower tail @f$P(x)@f$ is defined by the integral,
 * \f[
 *   P(x) = \int_{-\infty}^{x}\mathrm{d}x'p(x')
 * \f]
 * and gives the probability of a variate taking a value less than @f$x@f$.
 *
 * The cumulative distribution function for the uppper tail @f$Q(x)@f$ is defined by the integral,
 * \f[
 *   Q(x) = \int_{x}^{\infty}\mathrm{d}x'p(x')
 * \f]
 * and gives the probability of a variate taking a value greater than @f$x@f$.
 *
 * The upper and lower cumulative distribution functions are related by @f$P(x) + Q(x) = 1@f$, and satisfy @f$0 \leq P(x) \leq 1@f$, @f$0 \leq Q(x) \leq 1@f$.
 *
 * The inverse cumulative distributions, @f$x = P^{-1}(P)@f$ and @f$x = Q^{-1}(Q)@f$ give the values of @f$x@f$ which correspond to a specific value of @f$P@f$ or @f$Q@f$. They can be used to find confidence limits from probability function.
 */

#ifndef  PRECISION_H
#define  PRECISION_H

#include <immintrin.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>

#ifdef SINGLE_PRECISION
    #include <fftw3.h>
#else
    #include <fftw3.h>
#endif

#include "THUNDERConfig.h"
#include "Config.h"

#ifdef SINGLE_PRECISION

    /**
     * @brief single precision float
     */
    typedef float RFLOAT;

    /**
     * @brief complex in FFTW of single precision
     */
    #define TSFFTW_COMPLEX fftwf_complex

    /**
     * @brief plan in FFTW of single precision
     */
    #define TSFFTW_PLAN fftwf_plan

    /**
     * @brief number of bytes of single precision float in MPI
     */
    #define TS_MPI_DOUBLE MPI_FLOAT

    /**
     * @brief number of bytes of single precision complex in MPI
     */
    // #define TS_MPI_DOUBLE_COMPLEX MPI_COMPLEX
    // Some MPI environment do not support MPI_COMPLEX. It may appear fine during compilation, but go crash during execution.

    /**
     * @brief maximum possible value of single precision float
     */
    #define TS_MAX_RFLOAT_VALUE FLT_MAX


    /**
     * @brief a Complex object contains 2 single precision floats
     */
    typedef struct _complex_float_t
    {
        /**
         * @brief two single precision floats as the content of a Complex object
         */
        float dat[2];
    } Complex;

#else

    /**
     * @brief double precision float
     */
    typedef double RFLOAT;

    /**
     * @brief complex in FFTW of double precision
     */
    #define TSFFTW_COMPLEX fftw_complex

    /**
     * @brief plan in FFTW of double precision
     */
    #define TSFFTW_PLAN fftw_plan

    /**
     * @brief number of bytes of double precision float in MPI
     */
    #define TS_MPI_DOUBLE MPI_DOUBLE

    /**
     * @brief number of bytes of double precision complex in MPI
     */
    // #define TS_MPI_DOUBLE_COMPLEX MPI_DOUBLE_COMPLEX
    // Some MPI environment do not support MPI_DOUBLE_COMPLEX. It may appear fine during compilation, but go crash during execution.

    /**
     * @brief maximum possible value of double precision float
     */
    #define TS_MAX_RFLOAT_VALUE DBL_MAX

    /**
     * @brief a Complex object contains 2 double precision floats
     */
    typedef struct _complex_float_t
    {
        /**
         * @brief two double precision floats as the content of a Complex object
         */
        double dat[2];
    } Complex;

#endif

/**
 * @brief Calculate @f$x = Q^{-1}(Q)@f$, where the upper tail cumulative function @f$Q(x)@f$ of chi-squared distribution with degrees of freedom @f$\nu@f$ equals to @f$Q@f$ at @f$x@f$.
 *
 * @return @f$x@f$ where @f$Q(x) = Q@f$
 */
RFLOAT TSGSL_cdf_chisq_Qinv(const RFLOAT Q, /**< [in] uppper tail cumulative function value @f$Q@f$ */
                            const RFLOAT nu /**< [in] degree of freedom @f$\nu@f$ of chi-squared distribution */
                           );
/**
 * @brief Calculate @f$x = Q^{-1}(Q)@f$, where the upper tail cumulative function @f$Q(x)@f$ of Gaussian distribution with standard deviation @f$\sigma@f$ equals to @f$Q@f$ at @f$x@f$.
 *
 * @return @f$x@f$ where @f$Q(x) = Q@f$
 */
RFLOAT TSGSL_cdf_gaussian_Qinv(const RFLOAT Q,    /**< [in] upper tail cumulative function value @f$Q@f$ */
                               const RFLOAT sigma /**< [in] standard deviation @f$\sigma@f$ of Gaussian distribution */
                              );
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
void TSGSL_ran_bivariate_gaussian(const gsl_rng *r, /**< [in] */
                                  RFLOAT sigma_x,   /**< [in] */
                                  RFLOAT sigma_y,   /**< [in] */
                                  RFLOAT rho,       /**< [in] */
                                  RFLOAT *x,        /**< [out] */
                                  RFLOAT *y         /**< [out] */
                                 );
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
int TSGSL_fit_linear(const RFLOAT *x,      /**< [in] */
                     const size_t xstride, /**< [in] */
                     const RFLOAT *y,      /**< [in] */
                     const size_t ystride, /**< [in] */
                     const size_t n,       /**< [in] */
                     RFLOAT *c0,           /**< [out] */
                     RFLOAT *c1,           /**< [out] */
                     RFLOAT *cov00,        /**< [out] */
                     RFLOAT *cov01,        /**< [out] */
                     RFLOAT *cov11,        /**< [out] */
                     RFLOAT *sumsq         /**< [out] */
                    );
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
void TSGSL_ran_dir_2d (const gsl_rng * r, RFLOAT * x, RFLOAT * y);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_ran_flat (const gsl_rng * r, const RFLOAT a, const RFLOAT b);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_ran_gaussian (const gsl_rng * r, const RFLOAT sigma);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_sf_bessel_I0(const RFLOAT x);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_sf_bessel_In(const int n, const RFLOAT x);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_sf_bessel_Inu(RFLOAT nu, RFLOAT x);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_sf_bessel_j0(const RFLOAT x);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_sf_bessel_Jnu(const RFLOAT nu, const RFLOAT x);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_sf_sinc(const RFLOAT x);


/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
inline RFLOAT TS_FABS(const RFLOAT x)
{
#ifdef SINGLE_PRECISION
    return fabsf(x);
#else
    return fabs(x);
#endif
}

/**
 *  @brief Calculate the trigonometric sin of an angle x with type RFLOAT, which is either of single precision or double precision, determined at compiled time.
 *  
 *  @return the sin of the argument
 */
inline RFLOAT TS_SIN(const RFLOAT x /**< [in] the argument whose sin value is to be determined. */)
{
#ifdef SINGLE_PRECISION
    return sinf(x);
#else
    return sin(x);
#endif
}


/**
 *  @brief Calculate the trigonometric cosine of an angle x with type RFLOAT, which is either of single precision or double precision, determined at compiled time.
 *  
 *  @return the cosine of the argument
 */
inline RFLOAT TS_COS(const RFLOAT x /**< [in] the argument whose cosine value is to be determined */)
{
#ifdef SINGLE_PRECISION
    return cosf(x);
#else
    return cos(x);
#endif
}

/**
 *  @brief Calculate the correctly rounded positive square root of x, with type of RFLOAT, which is either single precision or double precision determined at compilied time.
 *  
 *  @return the positive square root of x
 */
inline RFLOAT TS_SQRT(const RFLOAT x /**< [in] the argument whose positive square root value is to be detemined. */)

{
#ifdef SINGLE_PRECISION
    return sqrtf(x);
#else
    return sqrt(x);
#endif
}


/**
 *  @brief Get the greater of two RFLOAT values. The type of RFLOAT is either float or double, determined at compilied time. 
 *
 *  @return the larger of a and b.
 */
inline RFLOAT TSGSL_MAX_RFLOAT(const RFLOAT a, /**< [in] an argument. */
                               const RFLOAT b  /**< [in] another argument. */
                              )
{
    return a > b ? a : b;
}

/**
 *  @brief Get the smaller of two RFLOAT values. The type of RFLOAT is either float or double, determined at compilied time. 
 *
 *  @return the smaller of a and b.
 */
inline RFLOAT TSGSL_MIN_RFLOAT(const RFLOAT a, /**< [in] an argument. */
                               const RFLOAT b  /**< [in] another argument. */
                              )
{
    return a < b ? a : b;
}

/**
 *  @brief Calculate the square of complex modulus length 
 *
 *  @return the square of modulus length of z
 */
RFLOAT TSGSL_complex_abs2 (Complex z /**< [in] the argument whose square of modulus length is determined*/);


/**
 *  @brief Determine whether x is infinity
 *
 *  @return 1 if x is positive infinity, -1 if x is negative infinity and 0 otherwise

 */
int TSGSL_isinf (const RFLOAT x /**< [in] the argment used to determined whether it is infinity. */);

/**
 *  @brief Determine whether x is a number
 *
 *  @return 1 if x is not-a-number and 0 otherwise
 */
int TSGSL_isnan (const RFLOAT x /**< [in] the argment used to determined whether it is a number. */);
;

/**
 *  @brief Calculate the value of @f$x^2@f$
 *
 *  @return The result @f$x^2@f$
 */
RFLOAT TSGSL_pow_2(const RFLOAT x /**< [in] the argument whose @f$x^2@f$ is calculated. */);

/**
 *  @brief Calculate the value of @f$x^3@f$
 *
 *  @return The result of @f$x^3@f$
 */
RFLOAT TSGSL_pow_3(const RFLOAT x /**< [in] the argument whose @f$x^3@f$ is calculated. */);

/**
 *  @brief Calculate the value of @f$x^4@f$
 *
 *  @return The result of @f$x^4@f$
 */
RFLOAT TSGSL_pow_4(const RFLOAT x /**< [in] the argument whose @f$x^4@f$ is calculated. */);

/**
 *  @brief Shuffle the data items in buffer base randomly.
 */
void TSGSL_ran_shuffle (const gsl_rng *r, /**< [in] randon engine used for randomness. */
                        void *base,       /**< [in,out] data elements to be shuffled. */
                        size_t nmembm,    /**< [in] number of data items to be shuffled. */
                        size_t size       /**< [in] size of each data items. */
                       );


/**
 *  @brief Create an instance of a random nunber generator
 *  
 *  @return a pointer to a newly-created instance of a random number generator of type T
 */
gsl_rng *TSGSL_rng_alloc (const gsl_rng_type *T /**< [in] type of random number generator*/);

/**
 *  @brief Free all the memory associated with the generator r
 *
 */
void TSGSL_rng_free (gsl_rng *r /**< [in] generator to be freed*/);

/**
 *  @brief Get a random integer from the generator r. The minimum and maximum values depend on the algorithm used, but all integers in the range [min, max] are equally likely. The values of min and max can be determined using the auxiliary functions gsl_rng_max() and gsl_rng_min()
 *
 *  @return a random integer from the generator r.
 */
size_t TSGSL_rng_get (const gsl_rng *r /**< [in] a random generator. */);

/**
 *  @brief Initialize (or “seeds”) the random number generator
 */
void TSGSL_rng_set (const gsl_rng *r /**< [in] random number generator*/,
                    size_t seed      /**< [in] seed used for random number generator. */
                   );


/**
 *  @brief Get a float point number with type of RFLOAT uniformly distributed in the range [0,1)
 *
 *  @return a floating point number with type of RFLOAT uniformly distributed in the range [0,1)
 */
RFLOAT TSGSL_rng_uniform (const gsl_rng * r /**< [in] a random engine used to generate the number. */);

/**
 *  @brief Get a random integer from 0 to @f$n-1@f$ inclusive by scaling down and/or discarding samples from the generator @f$r@f$.
 *
 *  @return A random integer in the range [0, n)
 */
size_t TSGSL_rng_uniform_int (const gsl_rng * r, /**< [in] a random engine used to generate the number. */
                              size_t n           /**< [in] the upper bound*/
                             );


/**
 * @brief Sort the @f$n@f$ elements of the array data with stride stride into ascending numerical order 
 */
void TSGSL_sort(RFLOAT * data,/**< [in,out] data elements to be sorted. */
                 const size_t stride /**< [in] stride argument */,
                 const size_t n      /**< [in] the number of elements to be sorted*/
                );

/**
 * @brief Copy the @f$k@f$ largest elements of the array @f$src@f$, of size @f$n@f$ and stride stride, in descending numerical order into the array @f$dest@f$. @f$k@f$ must be less than or equal to @f$n@f$. The data @f$src@f$ is not modified by this operation.
 *
 * @return 1 if copy success, or 0 if copy failed 
 */
int TSGSL_sort_largest(RFLOAT * dst,        /**< [out] destination buffer. */
                       const size_t k,      /**< [in]  number of elements to be copied. */
                       const RFLOAT * src,  /**< [in]  source buffer. */
                       const size_t stride, /**< [in]  stride argument. */
                       const size_t n       /**< [in]  total number of source elements. */
                      );


/**
 *  @brief Sort the @f$n@f$ elements of the array @f$src@f$ with stride stride into ascending order, storing the resulting permutation in @f$dst@f$. The elements of @f$dst@f$ give the index of the vector element. 
 */
void TSGSL_sort_index(size_t * dst,        /**< [out] destination buffer. */
                      const RFLOAT * src,  /**< [in]  source buffer. */
                      const size_t stride, /**< [in]  stride argument. */
                      const size_t n       /**< [in]  total number of source elements. */
                     );

        
/**
 *  @brief Store the indices of the @f$k@f$ smallest elements of the array @f$src@f$, of size @f$n@f$ and stride stride, in the array @f$p@f$.
 */
void TSGSL_sort_smallest_index(size_t* dst,         /**< [out] destination buffer. */               
                               const size_t k,      /**< [in]  number of elements to be choosed. */
                               const RFLOAT* src,   /**< [in]  source buffer. */
                               const size_t stride, /**< [in]  stride argument. */
                               const size_t n       /**< [in]  total number of source elements. */
                              );

/**
 *  @brief Store the indices of the @f$k@f$ largest elements of the array @f$src@f$, of size @f$n@f$ and stride stride, in the array @f$p@f$.
 */
void TSGSL_sort_largest_index(size_t* dst,          /**< [out] destination buffer. */               
                               const size_t k,      /**< [in]  number of elements to be choosed. */
                               const RFLOAT* src,   /**< [in]  source buffer. */
                               const size_t stride, /**< [in]  stride argument. */
                               const size_t n       /**< [in]  total number of source elements. */
                              );


/**
 *  @brief Get the maximum value in @f$data@f$, a dataset of length @f$n@f$ with stride stride.
 *
 *  @return the maximum value with stride stride.
 */
RFLOAT TSGSL_stats_max(const RFLOAT data[], /**< [in] source data buffer. */
                       const size_t stride, /**< [in] stride argument. */
                       const size_t n       /**< [in] total number of source elements. */
                      );
/**
 *  @brief Get the minimum value in @f$data@f$, a dataset of length @f$n@f$ with stride stride.
 *
 *  @return the minimum value with stride stride.
 */
RFLOAT TSGSL_stats_min(const RFLOAT data[],  /**< [in] source data buffer. */
                        const size_t stride, /**< [in] stride argument. */
                        const size_t n       /**< [in] total number of source elements. */
                      );
/**
 *  @brief Get the mean value in @f$data@f$, a dataset of length @f$n@f$ with stride stride.
 *
 *  @return the mean value with stride stride.
 */
RFLOAT TSGSL_stats_mean(const RFLOAT data[], /**< [in] source data buffer. */
                       const size_t stride,  /**< [in] stride argument. */
                       const size_t n        /**< [in] total number of source elements. */
                      );


/**
 *  @brief Get a quantile value of @f$sorted_data@f$, a double-precision array of length @f$n@f$ with stride stride. The elements of the array must be in ascending numerical order. The quantile is determined by the @f$f@f$, a fraction between 0 and 1.
 *
 *  @return a quantile value.
 */
RFLOAT TSGSL_stats_quantile_from_sorted_data(const RFLOAT sorted_data[], /**< [in] sorted data buffer to be searched. */ 
                                             const size_t stride,        /**< [in] stride argument. */
                                             const size_t n,             /**< [in] the length of the sorted data buffer. */ 
                                             const RFLOAT f              /**< [in] the argument determines the quantile. */
                                            );

/**
 *  @brief Get the standard deviation with stride stride.
 */
RFLOAT TSGSL_stats_sd(const RFLOAT data[], /**< [in] data elements buffer to be used. */
                      const size_t stride, /**< [in] the argument stride. */
                      const size_t n       /**< [in] the length of data elements buffer. */
                     );

/**
 *  @brief Get the standard mean deviation with stride stride.
 */
RFLOAT TSGSL_stats_sd_m(const RFLOAT data[], /**< [in] data elements buffer to be used. */
                        const size_t stride, /**< [in] the argument stride. */
                        const size_t n,      /**< [in] the length of data elements buffer. */
                        const RFLOAT mean    /**< [in] the mean value to be used. */
                       );

/**
 *  @brief Call FFTW interfaces in multi-threaded manner.
 */
int TSFFTW_init_threads();

/**
 *  @brief Free all memory and other resources allocated internally by FFTW.
 */
void TSFFTW_cleanup_threads();

/**
 *  @brief Deallocate FFTW plan.
 */
void TSFFTW_destroy_plan(TSFFTW_PLAN plan /**< [in] plan to be destroyed. */);

/**
 *  @brief Execute FFTW plan.
 */
void TSFFTW_execute(const TSFFTW_PLAN plan /**< [in] plan to be executed. */);

/**
 *  @brief Execute FFTW plan to compute the transform from real space to Fourier space. 
 */
void TSFFTW_execute_dft_r2c(const TSFFTW_PLAN p, /**< [in]  plan to be used. */
                            RFLOAT *in,          /**< [in]  data elements in real space to be transformed. */
                            TSFFTW_COMPLEX *out  /**< [out] result of transformation in Fourier space. */
                           );

/**
 *  @brief Execute FFTW plan to compute the transform from Fourier space to real space. 
 */
void TSFFTW_execute_dft_c2r(const TSFFTW_PLAN p, /**< [in]  plan to be used. */
                            TSFFTW_COMPLEX *in,  /**< [in]  data elements in Fourier space to be transformed. */
                            RFLOAT *out         /**<  [out] result of transformation in real space. */
                           ); 

/**
 *  @brief Allocate n bytes aligned memory. 
 *
 *  @return a pointer points to new allocated memory.
 */
void *TSFFTW_malloc(size_t n /**< [in] bytes of memory to be allocated. */);

/**
 *  @brief Free memory pointer @f$p@f$ points to. 
 */
void TSFFTW_free(void *p /**< [in] a pointer pointed to a memory which will be freed. */);

/**
 *  @brief Create a 2D fftw plan used for transformation from real space to fourier space.
 */
TSFFTW_PLAN TSFFTW_plan_dft_r2c_2d(int n0,              /**< [in] length of first dimension. */
                                   int n1,              /**< [in] length of second dimension. */
                                   RFLOAT *in,          /**< [in] data elements in real space. */
                                   TSFFTW_COMPLEX *out, /**< [out] data elements in fourier space. */
                                   unsigned flags       /**< [in] flags used for control transformation. */
                                  );

/**
 *  @brief Create a 3D fftw plan used for transformation from real space to fourier space.
 */
TSFFTW_PLAN TSFFTW_plan_dft_r2c_3d(int n0,              /**< [in] length of first dimension. */
                                   int n1,              /**< [in] length of second dimension. */
                                   int n2,              /**< [in] length of third dimension. */
                                   RFLOAT *in,          /**< [in] data elements in real space. */
                                   TSFFTW_COMPLEX *out, /**< [out] data elements in fourier space. */
                                   unsigned flags       /**< [in] flags used for control transformation. */
                                  );

/**
 *  @brief Create a 2D fftw plan used for transformation from fourier space to real space.
 */
TSFFTW_PLAN TSFFTW_plan_dft_c2r_2d(int n0,              /**< [in] length of first dimension. */
                                   int n1,              /**< [in] length of second dimension. */
                                   TSFFTW_COMPLEX *in,  /**< [in] data elements in fourier space. */
                                   RFLOAT *out,        /**< [out] data elements in real space. */
                                   unsigned flags       /**< [in] flags used for control transformation. */
                                  );

/**
 *  @brief Create a 3D fftw plan used for transformation from fourier space to real space.
 */
TSFFTW_PLAN TSFFTW_plan_dft_c2r_3d(int n0,             /**< [in] length of first dimension. */
                                   int n1,             /**< [in] length of second dimension. */
                                   int n2,             /**< [in] length of third dimension. */
                                   TSFFTW_COMPLEX *in, /**< [in] data elements in fourier space. */
                                   RFLOAT *out,        /**< [out] data elements in real space. */
                                   unsigned flags      /**< [in] flags used for control transformation. */
                                  );

/**
 *  @brief Initialized thread number used for planner routines.
 */
void TSFFTW_plan_with_nthreads(int nthreads /**< [in] thread number used for planner routines. */);

/**
 *  @brief Use for instructs FFTW to spend at most @f$seconds@f$ seconds (approximately) in the planner.
 */
void TSFFTW_set_timelimit(RFLOAT seconds /**< [in] max seconds to spend. */);

#endif // PRECISION_H
