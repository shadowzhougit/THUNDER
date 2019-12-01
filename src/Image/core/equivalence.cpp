/** @file
 *  @author Mingxu Hu
 *  @version 1.4.14.190713
 *  @copyright GPLv2
 *
 *  ChangeLog
 *  AUTHOR     | TIME       | VERSION       | DESCRIPTION
 *  ------     | ----       | -------       | -----------
 *  Mingxu Hu  | 2019/07/13 | 1.4.14.190713 | new file
 *
 */
#include "core/equivalence.h"

bool operator==(const Image& i1,
                const Image& i2)
{
    if (&i1 == &i2)
    {
        return true;
    }

    if (i1._nCol != i2._nCol)
    {
        return false;
    }

    if (i1._nRow != i2._nRow)
    {
        return false;
    }

    if (i1._dataRL == NULL && i2._dataRL != NULL)
    {
        return false;
    }

    if (i1._dataRL != NULL && i2._dataRL == NULL)
    {
        return false;
    }

    if (i1._dataRL != NULL && i2._dataRL != NULL)
    {
        for (size_t i = 0; i < i1._sizeRL; i++)
        {
            if (TS_FABS(i1._dataRL[i] - i2._dataRL[i]) > EQUAL_ACCURACY)
            {
                return false;
            }
        }
    }

    if (i1._dataFT == NULL && i2._dataFT != NULL)
    {
        return false;
    }

    if (i1._dataFT != NULL && i2._dataFT == NULL)
    {
        return false;
    }

    if (i1._dataFT != NULL && i2._dataFT != NULL)
    {
        for (size_t i = 0; i < i1._sizeFT; i++)
        {
            if (ABS(i1._dataFT[i] - i2._dataFT[i]) > EQUAL_ACCURACY)
            {
                return false;
            }
        }
    }

    return true;
}
