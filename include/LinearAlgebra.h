/** @file
 *  @author Mingxu Hu
 *  @version 1.4.14.190629
 *  @copyright GPLv2
 *
 *  ChangeLog
 *  AUTHOR     | TIME       | VERSION       | DESCRIPTION
 *  ------     | ----       | -------       | -----------
 *  Mingxu Hu  | 2019/06/29 | 1.4.14.190629 | new file
 *
 *  @brief 
 *
 */

#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include "THUNDERConfig.h"
#include "Macro.h"
#include "Precision.h"
#include "Typedef.h"

inline bool operator==(const dvec4& v1,
                       const dvec4& v2)
{
    for (int i = 0; i < 4; i++)
    {
        if (fabs(v1[i] - v2[i]) > EQUAL_ACCURACY)
            return false;
    }

    return true;
}

inline double dot(const dvec4& v1,
                  const dvec4& v2)
{
    double d = 0;

    for (int i = 0; i < 4; i++)
    {
        d += v1[i] * v2[i];
    }

    return d;
}

inline bool operator==(const dmat44& m1,
                       const dmat44& m2)
{
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
        {
            if (fabs(m1(i, j) - m2(i, j)) > EQUAL_ACCURACY)
                return false;
        }

    return true;
}

dmat44 tensor(const dvec4& v1,
              const dvec4& v2);

#endif // LINEAR_ALGEBRA_H
