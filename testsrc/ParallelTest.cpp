/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Volume.h"

#include "Parallel.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    MPI_Init(&argc, &argv);

    Parallel par;
    par.setMPIEnv();
    display(par);

    Volume vol(760, 760, 760, FT_SPACE);
    //Volume vol(100, 100, 100, FT_SPACE);
    SET_1_FT(vol);

    for (int i = 0; i < atoi(argv[1]); i++)
    {
        if (par.commRank() != MASTER_ID)
        {
            if (par.commRank() == HEMI_A_LEAD)
                CLOG(INFO, "LOGGER_SYS") << "HEMI_A: Round " << i;
            if (par.commRank() == HEMI_B_LEAD)
                CLOG(INFO, "LOGGER_SYS") << "HEMI_B: Round " << i;

            MPI_Allreduce_Large(MPI_IN_PLACE,
                                &vol[0],
                                vol.sizeFT(),
                                MPI_DOUBLE_COMPLEX,
                                MPI_SUM,
                                par.hemi());

            if (par.commRank() == HEMI_A_LEAD)
            {
                for (size_t j = 0; j < vol.sizeFT(); j++)
                    if ((REAL(vol[j]) != 2) ||
                        (IMAG(vol[j]) != 0))
                        {
                            cout << "Error!" << endl;
                            cout << j << " : ( " << REAL(vol[j])
                                 << ", " << IMAG(vol[j]) << endl;
                            break;
                        }
            }
            cout << endl;
        }
    }
    
    MPI_Finalize();
}
