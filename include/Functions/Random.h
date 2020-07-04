/*******************************************************************************
 * Author: Siyuan Ren, Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef RANDOM_H
#define RANDOM_H

#include <stdexcept>

#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>

#include <sys/types.h>
#include <sys/stat.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "Logging.h"
#include "Precision.h"
#include "Typedef.h"
#include "Functions.h"

gsl_rng* get_random_engine();

uvec shuffledIndex(const size_t n);

int drawWithWeightIndex(const vec& v);

#endif // RANDOM_H
