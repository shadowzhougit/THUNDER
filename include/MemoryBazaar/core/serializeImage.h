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

#ifndef MEMORY_BAZAAR_CORE_SERIALIZE_IMAGE_H
#define MEMORY_BAZAAR_CORE_SERIALIZE_IMAGE_H

#include "THUNDERConfig.h"
#include "Macro.h"
#include "Precision.h"
#include "Typedef.h"
#include "Image.h"
#include "Functions.h"

/**
 * _sizeRL
 * _sizeFT
 * RL or FT (true for RL, false for FT, bool)
 * _dataRL / _dataFT
 * _nCol (long)
 * _nRow (long)
 * _nColFT (long)
 * _box[2][2] (4 * size_t)
 */

size_t serializeSize(const Image& i)
{
    return 2 * sizeof(size_t) // _sizeRL + _sizeFT
         + sizeof(int) // -1, not RL or FT, 0, RL, 1, FT
         //+ sizeof(RFLOAT*)
         //+ sizeof(Complex*)
         + 2 * i._sizeFT * sizeof(RFLOAT) // _dataRL / _dataFT
         + 3 * sizeof(long) // _nCol + _nRow + _nColFT
         + 4 * sizeof(size_t); // _box[2][2]
};

void serialize(void* m,
               const Image& i)
{
    char* p = (char*)m;
    
    memcpy(p, &i._sizeRL, sizeof(size_t));

    p += sizeof(size_t);

    memcpy(p, &i._sizeFT, sizeof(size_t));

    p += sizeof(size_t);

    int flag = -1;

    if (i._dataRL != NULL)
    {
        flag = 0;
    }

    if (i._dataFT != NULL)
    {
        flag = 1;
    }

    memcpy(p, &flag, sizeof(int));

    p += sizeof(int);

    memcpy(p, &i._nCol, sizeof(long));

    p += sizeof(long);

    memcpy(p, &i._nRow, sizeof(long));

    p += sizeof(long);

    memcpy(p, &i._nColFT, sizeof(long));

    p += sizeof(long);
    
    memcpy(p, i._box, 4 * sizeof(size_t));

    p += 4 * sizeof(size_t);

    if (i._dataRL != NULL)
    {
        memcpy(p, i._dataRL, i._sizeRL * sizeof(RFLOAT));
    }

    if (i._dataFT != NULL)
    {
        memcpy(p, i._dataFT, 2 * i._sizeFT * sizeof(RFLOAT));
    }
};

void deserialize(Image& i,
                 const void* m,
                 const size_t maxLength)
{
    i.clear();

    char* p = (char*)m;
    
    memcpy(&i._sizeRL, p, sizeof(size_t));

    p += sizeof(size_t);

    memcpy(&i._sizeFT, p, sizeof(size_t));

    p += sizeof(size_t);

    int flag;
    memcpy(&flag, p, sizeof(int));

    p += sizeof(int);

    memcpy(&i._nCol, p, sizeof(long));

    p += sizeof(long);

    memcpy(&i._nRow, p, sizeof(long));

    p += sizeof(long);

    memcpy(&i._nColFT, p, sizeof(long));

    p += sizeof(long);
    
    memcpy(i._box, p, 4 * sizeof(size_t));

    p += 4 * sizeof(size_t);

    if ((flag == 0) && ((p - (char*)m) + i._sizeRL * sizeof(RFLOAT)) <= maxLength)
    {
        i._dataRL = (RFLOAT*)TSFFTW_malloc(i._sizeRL * sizeof(RFLOAT));

        memcpy(i._dataRL, p, i._sizeRL * sizeof(RFLOAT));
    }
    else
    {
        i._dataRL = NULL;
    }

    if ((flag == 1) && ((p - (char*)m) + 2 * i._sizeFT * sizeof(RFLOAT)) <= maxLength)
    {
        i._dataFT = (Complex*)TSFFTW_malloc(2 * i._sizeFT * sizeof(RFLOAT));

        memcpy(i._dataFT, p, 2 * i._sizeFT * sizeof(RFLOAT));
    }
    else
    {
        i._dataFT = NULL;
    }
};

#endif // MEMORY_BAZAAR_CORE_SERIALIZE_IMAGE_H
