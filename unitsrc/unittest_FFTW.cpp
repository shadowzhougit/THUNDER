/** @file
 *  @author Mingxu Hu 
 *  @version 1.4.11.081102
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 */

#include <gtest/gtest.h>

#include "ImageFile.h"

#include "FFT.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    loggerInit(argc, argv);

    Volume vol(100, 200, 300, RL_SPACE);

    std::cout << "sizeRL of vol = " << vol.sizeRL() << std::endl;

    VOLUME_FOR_EACH_PIXEL_RL(vol)
    {
        if (NORM_3(i, j, k) < 50)
        {
            vol.setRL(1, i, j, k);
        }
        else
        {
            vol.setRL(0, i, j, k);
        }
    }

    ImageFile imf;
    imf.readMetaData(vol);
    imf.writeVolume("before.mrc", vol, 1.32);

    FFT fft;

    fft.fw(vol, 10);
    fft.bw(vol, 10);

    imf.readMetaData(vol);
    imf.writeVolume("after.mrc", vol, 1.32);
}
