/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <fstream>

#include "Projector.h"
#include "ImageFile.h"
#include "FFT.h"

#define PF 2

using namespace std;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    Volume obj;

    std::cout << "Reading in Object" << std::endl;

    ImageFile imf(argv[1], "r");
    imf.readMetaData();
    imf.readVolume(obj);

    std::cout << "Size: " << obj.nColRL() << " x "
                     << obj.nRowRL() << " x "
                     << obj.nSlcRL() << std::endl;

    int N = obj.nColRL();

    std::cout << "Padding" << std::endl;

    Volume padObj;
    VOL_PAD_RL(padObj, obj, 2);
    
    std::cout << "Performing Fourier Transform" << std::endl;

    FFT fft;
    fft.fw(padObj);

    Projector projector;
    projector.setPf(PF);
    projector.setProjectee(padObj.copyVolume());

    char name[FILE_NAME_LENGTH];
    int counter = 0;

    Image image(N, N, RL_SPACE);
    
    ifstream fin(argv[2], ios::in);
    char line[1024];
    double phi, theta, psi, x, y;

    ImageFile oimf;

    while (fin.getline(line, sizeof(line)))
    {
        stringstream word(line);
        word >> phi;
        word >> theta;
        word >> psi;
        word >> x;
        word >> y;

        std::cout << "phi = " << phi
             << ", theta = " << theta
             << ", psi = " << psi
             << ", x = " << x
             << ", y = " << y
             << std::endl;

        fft.fw(image);
        SET_0_FT(image);
        projector.project(image, phi, theta, psi, x, y);
        fft.bw(image);

        sprintf(name, "%08d.bmp", counter);
        image.saveRLToBMP(name);

        sprintf(name, "%08d.mrc", counter);
        oimf.readMetaData(image);
        oimf.writeImage(name, image);

        counter++;
    }

    fin.clear();
    fin.close();

    return 0;
}
