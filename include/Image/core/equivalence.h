/** @file
 *  @author Mingxu Hu
 *  @version 1.4.14.190629
 *  @copyright GPLv2
 *
 *  ChangeLog
 *  AUTHOR     | TIME       | VERSION       | DESCRIPTION
 *  ------     | ----       | -------       | -----------
 *  Mingxu Hu  | 2019/06/29 | 1.4.14.190713 | new file
 *
 *  @brief 
 *
 */

#ifndef IMAGE_EQUIVALENCE_H
#define IMAGE_EQUIVALENCE_H

#include "THUNDERConfig.h"
#include "Macro.h"
#include "Precision.h"
#include "Typedef.h"
#include "Image.h"

bool operator==(const Image& i1,
                const Image& i2);

#endif // IMAGE_EQUIVALENCE_H
