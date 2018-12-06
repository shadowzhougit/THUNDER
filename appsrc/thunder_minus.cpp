/** @file
 *  @author Mingxu Hu
 *  @author Shouqing Li
 *  @version 1.4.11.081105
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu   Hu | 2015/03/23 | 0.0.1.050323  | new file
 *  Shouqing Li | 2018/09/28 | 1.4.11.080928 | add options 
 *  Mingxu   Hu | 2018/11/05 | 1.4.11.081105 | change this file to minus reference
 */

#include <fstream>
#include <iostream>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>

#include "ImageFile.h"
#include "Volume.h"

INITIALIZE_EASYLOGGINGPP

#define PROGRAM_NAME "thunder_minus"

#define emit_try_help() \
do \
    { \
        fprintf(stderr, "Try '%s --help' for more information.\n", \
                PROGRAM_NAME); \
    } \
while (0)

#define HELP_OPTION_DESCRIPTION "--help     display this help\n"

void usage(int status)
{
    if (status != EXIT_SUCCESS)
    {
        emit_try_help ();
    }
    else
    {
        printf("Usage: %s [OPTION]...\n", PROGRAM_NAME);

        fputs("Read two input image-files and output the diff file of their pixels' value.\n", stdout);

        fputs("-o    set the directory of output file.\n", stdout);
        fputs("--inputA    set the directory of input file A.\n", stdout);
        fputs("--inputB    set the directory of input file B.\n", stdout);
        fputs("--pixelsize    set the pixelsize.\n", stdout);

        fputs(HELP_OPTION_DESCRIPTION, stdout);

        fputs("Note: all parameters are indispensable.\n", stdout);

    }
    exit(status);
}

static const struct option long_options[] = 
{
    {"inputA", required_argument, NULL, 'a'},
    {"inputB", required_argument, NULL, 'b'},
    {"pixelsize", required_argument, NULL, 'p'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
};

int main(int argc, char* argv[])
{
    
    int opt;
    char* output;
    char* inputA;
    char* inputB;
    double pixelsize;

    int option_index = 0;

    if(optind == argc)
    {
        usage(EXIT_FAILURE);
    }

    while((opt = getopt_long(argc, argv, "o:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
            case('o'):
                output = optarg;
                break;
            case('a'):
                inputA = optarg;
                break;
            case('b'):
                inputB = optarg;
                break;
            case('p'):
                pixelsize = atof(optarg);
                break;
            case('h'):
                usage(EXIT_SUCCESS);
                break;
            default:
                usage(EXIT_FAILURE);
        }
    }

    loggerInit(argc, argv);

    CLOG(INFO, "LOGGER_SYS") << "Reading Map";

    ImageFile imfA(inputA, "rb");
    imfA.readMetaData();

    Volume refA;
    imfA.readVolume(refA);

    ImageFile imfB(inputB, "rb");
    imfB.readMetaData();

    Volume refB;
    imfB.readVolume(refB);

    FOR_EACH_PIXEL_RL(refA)
    {
        refA(i) -= refB(i);
    }

    ImageFile imf;
    imf.readMetaData(refA);
    imf.writeVolume(output, refA, pixelsize);

    return 0;
}
