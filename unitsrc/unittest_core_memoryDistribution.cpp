/** @file
 *  @author Mingxu Hu 
 *  @version 1.4.11.081102
 *  @copyright GPLv2
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2019/06/29 | 1.4.14.090629 | new file
 */

#include "core/serializeImage.h"
#include "core/memoryDistribution.h"

#include "Image.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    MemoryDistribution md;

    for (size_t i = 2; i < pow(2, 8); i++)
    {
        assignMemoryDistribution(md, i * GIGABYTE, serializeSize(Image(320, 320, RL_SPACE)), 1, 131319, 3 * 24);

        // std::cout << "Memory Usage = " << i << " GB" << std::endl;
        // std::cout << "md.nStallImg = " << md.nStallImg << std::endl;
        // std::cout << "md.nStallDatPR = " << md.nStallDatPR << std::endl;

        std::cout << i << ", " << md.nStallImg * 4 * (RFLOAT)serializeSize(Image(320, 320, RL_SPACE)) << ", " << md.nStallDatPR * 4 * M_PI / 8 * (RFLOAT)serializeSize(Image(320, 320, RL_SPACE)) << std::endl;
    }
}
