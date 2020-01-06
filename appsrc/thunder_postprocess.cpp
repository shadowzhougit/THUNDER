/** @file
 *  @author Mingxu Hu
 *  @author Shouqing Li
 *  @version 1.4.11.080928
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu   Hu | 2015/03/23 | 0.0.1.050323  | new file
 *  Shouqing Li | 2018/09/28 | 1.4.11.080928 | add options
 *  Shouqing Li | 2018/01/07 | 1.4.11.090107 | output error information of missing options
 *
 *  @brief thunder_postprocess.cpp helps users to post-process the input image-file. The parameters provided by users are the directory of two parts input files, radius of mask, number of threads and pixelsize.
 *
 */

#include <iostream>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>
#include <iostream>

#include "Postprocess.h"
#include "Utils.h"

using namespace std;

INITIALIZE_EASYLOGGINGPP

#define PROGRAM_NAME "thunder_postprocess"

#define emit_try_help() \
do \
    { \
        fprintf(stderr, "Try '%s --help' for more information.\n", \
                PROGRAM_NAME); \
    } \
while(0)

void usage (int status)
{
    if (status != EXIT_SUCCESS)
    {
        emit_try_help ();
    }
    else
    {
        printf("Usage: %s [OPTION]...\n", PROGRAM_NAME);

        fputs("\nPost-process the input image-file.\n\n", stdout);

        fputs("--inputA       set the filename of input file A.\n", stdout);
        fputs("--inputB       set the filename of input file B.\n", stdout);
        fputs("--mask         set the filename of mask file.\n", stdout);
        fputs("--prefix       set the prefix of the output files.\n", stdout);
        fputs("--pixelsize    set the pixelsize.\n", stdout);
        fputs("-j             set the number of threads to carry out work.\n", stdout);

        fputs("\n--help         display this help\n", stdout);
        fputs("Note: all parameters are indispensable.\n", stdout);
    }
    exit(status);
}

static const struct option long_options[] =
{
    {"inputA", required_argument, NULL, 'a'},
    {"inputB", required_argument, NULL, 'b'},
    {"mask", required_argument, NULL, 'm'},
    {"prefix", required_argument, NULL, 'p'},
    {"pixelsize", required_argument, NULL, 'x'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
};

int main(int argc, char* argv[])
{

    int opt;
    char* inputA;
    char* inputB;
    char* mask;
    char* prefix;
    double pixelsize;
    int nThread;

    char option[6] = {'m', 'a', 'b', 'p', 'j', 'x'};

    int option_index = 0;

    if(optind == argc)
    {
        usage(EXIT_SUCCESS);
    }

    while((opt = getopt_long(argc, argv, "j:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
            case('m'):
                mask = optarg;
                option[0] = '\0';
                break;
            case('a'):
                inputA = optarg;
                option[1] = '\0';
                break;
            case('b'):
                inputB = optarg;
                option[2] = '\0';
                break;
            case('p'):
                prefix = optarg;
                option[3] = '\0';
                break;
            case('x'):
                pixelsize = atof(optarg);
                option[4] = '\0';
                break;
            case('j'):
                nThread = atoi(optarg);
                option[5] = '\0';
                break;
            case('h'):
                usage(EXIT_SUCCESS);
                break;
            default:
                usage(EXIT_FAILURE);
        }
    }

    optionCheck(option, sizeof(option) / sizeof(*option), long_options);

    loggerInit(argc, argv);

    TSFFTW_init_threads();

    omp_set_nested(false);

    Postprocess pp(inputA,
                   inputB,
                   mask,
                   prefix,
                   pixelsize);

    pp.run(nThread);

    TSFFTW_cleanup_threads();

    return 0;
}