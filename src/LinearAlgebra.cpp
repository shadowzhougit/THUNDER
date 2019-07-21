/** @file
 *  @author Mingxu Hu
 *  @version 1.4.14.190629
 *  @copyright GPLv2
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2019/06/29 | 1.4.14.190629 | new file
 */

#include "LinearAlgebra.h"

dmat44 tensor(const dvec4& v1,
              const dvec4& v2)
{
    dmat44 tensor;

    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
        {
            tensor(i, j) = v1(i) * v2(j);
        }

    return tensor;
}
