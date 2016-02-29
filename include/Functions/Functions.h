/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cmath>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>

#define MAX(a, b) GSL_MAX(a, b)

#define MAX_3(a, b, c) MAX(MAX(a, b), c)

#define MIN(a, b) GSL_MIN(a, b)

#define MIN_3(a, b, c) MIN(MIN(a, b), c)

#define NORM(a, b) sqrt(gsl_pow_2(a) + gsl_pow_2(b))

#define NORM_3(a, b, c) sqrt(gsl_pow_2(a) + gsl_pow_2(b) + gsl_pow_2(c))

void normalise(gsl_vector& vec);

double MKB_FT(const double r,
              const double a,
              const double alpha);
/* Modified Kaiser Bessel Function, m = 2 */
/* Typically, a = 2.0 and alpha = 3.6 */

double MKB_RL(const double r,
              const double a,
              const double alpha);
/* Inverse Fourier Transform of Modified Kaiser Bessel Function, m = 2, n = 3 */
/* Typically, a = 2.0 and alpha = 3.6 */

#endif // FUNCTIONS_H
