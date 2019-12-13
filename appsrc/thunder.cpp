/** @file
 *  @author Mingxu Hu
 *  @author Shouqing Li
 *  @version 1.4.11.081025
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2015/03/23 | 0.0.1.050323  | new file
 *  Shouqing Li | 2018/10/25 | 1.4.11.081025 | add test for directory
 *  Mingxu Hu   | 2018/10/30 | 1.4.11.081030 | solve conflict during merging
 *
 *  @brief thunder.cpp initiates the MPI, following the completion of reading and logging the json files. And according to the set parameters, thunder.cpp will carry out computation chosen from three models,namely 2D classification, 3D classification and 3D refinement. In the final, the results will be exported to the file wrote in json.
 *
 */

#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <stdlib.h>
#include <errno.h>
#include <json/json.h>
#include <unistd.h>

#include "Config.h"
#include "Logging.h"
#include "Macro.h"
#include "Projector.h"
#include "Reconstructor.h"
#include "FFT.h"
#include "ImageFile.h"
#include "Particle.h"
#include "CTF.h"
#include "Optimiser.h"

using namespace std;

//int mkDir(string s, mode_t mode)
//{
//    size_t pos = 0;
//    string dir;
//    int mdRet = 0;
//    while ((pos = s.find_first_of('/', pos)) != string::npos)
//    {
//        dir = s.substr(0, pos++);

//        if (dir.size() == 0)
//        {
//            continue;
//        }

//        if ((mdRet = mkdir(dir.c_str(), mode)) && errno != EEXIST)
//        {
//            return mdRet;
//        }

//    }

//    return mdRet;
//}


void mkDir(string s, mode_t mode)
{
    size_t pos = 0;
    string dir;
    int mdRet = 0;
    while ((pos = s.find_first_of('/', pos)) != string::npos)
    {
        dir = s.substr(0, pos++);

        if (dir.size() == 0)
        {
            continue;
        }

        if ((mdRet = mkdir(dir.c_str(), mode)) && errno != EEXIST)
        {
            return;
        }
    }

}


int createCacheDirctory(OptimiserPara &thunderPara)
{
    if (thunderPara.cacheDirectory != NULL)
    {
        int len = strlen(thunderPara.cacheDirectory);
        if (len > 0)
        {
            if (thunderPara.cacheDirectory[len - 1] != '/' )
            {
                if (len < (FILE_NAME_LENGTH - 1))
                {
                    thunderPara.cacheDirectory[len] = '/';
                }

                else
                {
                    printf("Length of CacheDirectory[%d] is too long, maximum is: %d\n", len, FILE_NAME_LENGTH);
                    abort();
                }
            }
        }

        mkDir(string(thunderPara.cacheDirectory), 0755);
    }
    struct stat st;
    stat(thunderPara.cacheDirectory, &st);
    if(S_ISDIR(st.st_mode))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

inline Json::Value JSONCPP_READ_ERROR_HANDLER(const Json::Value src,
                                              const std::string basicClass,
                                              const std::string optionKey
                                             )
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (src[basicClass] == Json::nullValue)
    {
        //CLOG(FATAL, "LOGGER_SYS") << "Json parameter file BASIC CLASS \""
        //                          << basicClass
        //                          << "\" is not exit. Please make sure of it. ";

        /**
         *  Note: This function is called before initLogger, so we should not use aboved log function in advanced.
         */
        if (rank == 0)
        {
            fprintf(stderr, "\n  ERROR: Json parameter file BASIC CLASS \"%s\" does not exist, please check it.\n", basicClass.c_str());
        }

        abort();
    }

    else if (src[basicClass][optionKey] == Json::nullValue)
    {
        if (rank == 0)
        {
            fprintf(stderr, "\n  ERROR: Json parameter file KEY \"%s\" in BASIC CLASS \"%s\" does not exist, please check it\n", optionKey.c_str(), basicClass.c_str());
        }

        abort();
    }

    else
    {
        return src[basicClass][optionKey];
    }
}

/**
 *  This function is added by huabin
 *  This function is used to covert seconds to day:hour:min:sec format

 void fmt_time(int timeInSeconds, char *outputBuffer)
 {
 int day = 0;
 int hour = 0;
 int min = 0;
 int sec = 0;
 int inputSeconds = timeInSeconds;

 day = timeInSeconds / (24 * 3600);
 timeInSeconds = timeInSeconds % (24 * 3600);
 hour = timeInSeconds/3600;
 timeInSeconds = timeInSeconds%3600;
 min = timeInSeconds/60;
 timeInSeconds = timeInSeconds%60;
 sec = timeInSeconds;
 snprintf(outputBuffer, 512, "%ds (%d days:%d hours:%d mins:%d seconds)\n", inputSeconds, day, hour, min, sec);
 }
 ***/

template <size_t N>
static inline void copy_string(char (&array)[N], const std::string &source)
{
    if (source.size() + 1 >= N)
    {
        CLOG(FATAL, "LOGGER_SYS") << "String too large to fit in parameter. "
                                  << "Destination length is "
                                  << N
                                  << ", while source length is "
                                  << source.size() + 1;
        abort();
        return;
    }

    memcpy(array, source.c_str(), source.size() + 1);
}

void readPara(OptimiserPara &dst, const Json::Value src)
{
    dst.nThreadsPerProcess = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_N_THREADS_PER_PROCESS).asInt();
    dst.maximumMemoryUsagePerProcessGB = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_MAXIMUM_MEMORY_USAGE_PER_PROCESS_IN_GB).asFloat();

    if (JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_MODE).asString() == "2D")
    {
        dst.mode = MODE_2D;
    }

    else if (JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_MODE).asString() == "3D")
    {
        dst.mode = MODE_3D;
    }

    else
    {
        //cout<<__FILE__<<" , " << __LINE__ <<", " << __FUNCTION__ << ": " << "INEXISTENT NODE" <<endl;
        fprintf(stderr, "%s, %d, %s: INEXISTENT MODE\n", __FILE__, __LINE__, __FUNCTION__);
        abort();
    }

    dst.gSearch = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_G_SEARCH).asBool();
    dst.lSearch = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_L_SEARCH).asBool();
    dst.cSearch = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_C_SEARCH).asBool();
    dst.k = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_K).asInt();
    dst.size = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_SIZE).asInt();
    dst.pixelSize = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_PIXEL_SIZE).asFloat();
    dst.maskRadius = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_MASK_RADIUS).asFloat();
    dst.transS = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_TRANS_S).asFloat();
    dst.initRes = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_INIT_RES).asFloat();
    dst.globalSearchRes = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_GLOBAL_SEARCH_RES).asFloat();
    copy_string(dst.sym, JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_SYM).asString());
    copy_string(dst.initModel, JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_INIT_MODEL).asString());
    copy_string(dst.db, JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_DB).asString());
    copy_string(dst.parPrefix, JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_PAR_PREFIX).asString());
    //copy_string(dst.dstPrefix, JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_DST_PREFIX).asString());
    copy_string(dst.outputDirectory, JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_OUTPUT_DIRECTORY).asString());
    copy_string(dst.outputFilePrefix, JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_OUTPUT_FILE_PREFIX).asString());
    dst.coreFSC = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_CORE_FSC).asBool();
    dst.maskFSC = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_MASK_FSC).asBool();
    dst.parGra = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_PAR_GRA).asBool();
    dst.refAutoRecentre = JSONCPP_READ_ERROR_HANDLER(src, "Basic", KEY_REF_AUTO_RECENTRE).asBool();
    dst.performMask = JSONCPP_READ_ERROR_HANDLER(src, "Reference Mask", KEY_PERFORM_MASK).asBool();
    dst.globalMask = JSONCPP_READ_ERROR_HANDLER(src, "Reference Mask", KEY_GLOBAL_MASK).asBool();
    copy_string(dst.mask, JSONCPP_READ_ERROR_HANDLER(src, "Reference Mask", KEY_MASK).asString());
    dst.subtract = JSONCPP_READ_ERROR_HANDLER(src, "Subtract", KEY_SUBTRACT).asBool();
    copy_string(dst.regionCentre, JSONCPP_READ_ERROR_HANDLER(src, "Subtract", KEY_REGION_CENTRE).asString());
    dst.symmetrySubtract = JSONCPP_READ_ERROR_HANDLER(src, "Subtract", KEY_SYMMETRY_SUBTRACT).asBool();
    dst.reboxSize = JSONCPP_READ_ERROR_HANDLER(src, "Subtract", KEY_REBOX_SIZE).asInt();

    copy_string(dst.cacheDirectory, JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_CACHE_DIRECTORY).asString());
    dst.iterMax = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_ITER_MAX).asInt();
    dst.goldenStandard = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_GOLDEN_STANDARD).asBool();
    dst.pf = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_PF).asInt();
    dst.a = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_A).asFloat();
    dst.alpha = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_ALPHA).asFloat();

    if (dst.mode == MODE_2D)
    {
        dst.mS = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_M_S_2D).asInt();
        dst.mLR = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_M_L_R_2D).asInt();
    }

    else if (dst.mode == MODE_3D)
    {
        dst.mS = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_M_S_3D).asInt();
        dst.mLR = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_M_L_R_3D).asInt();
    }

    else
    {
        fprintf(stderr, "INEXISTENT MODE");
        abort();
    }

    dst.saveRefEachIter = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_SAVE_REF_EACH_ITER).asBool();
    dst.saveTHUEachIter = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_SAVE_THU_EACH_ITER).asBool();
    dst.mLT = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_M_L_T).asInt();
    dst.mLD = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_M_L_D).asInt();
    dst.mReco = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_M_RECO).asInt();
    dst.ignoreRes = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_IGNORE_RES).asFloat();
    dst.sclCorRes = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_SCL_COR_RES).asFloat();
    dst.thresCutoffFSC = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_THRES_CUTOFF_FSC).asFloat();
    dst.thresReportFSC = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_THRES_REPORT_FSC).asFloat();
    dst.thresSclCorFSC = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_THRES_SCL_COR_FSC).asFloat();
    dst.groupSig = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_GROUP_SIG).asBool();
    dst.groupScl = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_GROUP_SCL).asBool();
    dst.zeroMask = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_ZERO_MASK).asBool();
    dst.ctfRefineS = JSONCPP_READ_ERROR_HANDLER(src, "Advanced", KEY_CTF_REFINE_S).asFloat();
    dst.transSearchFactor = JSONCPP_READ_ERROR_HANDLER(src, "Professional", KEY_TRANS_SEARCH_FACTOR).asFloat();
    dst.perturbFactorL = JSONCPP_READ_ERROR_HANDLER(src, "Professional", KEY_PERTURB_FACTOR_L).asFloat();
    dst.perturbFactorSGlobal = JSONCPP_READ_ERROR_HANDLER(src, "Professional", KEY_PERTURB_FACTOR_S_GLOBAL).asFloat();
    dst.perturbFactorSLocal = JSONCPP_READ_ERROR_HANDLER(src, "Professional", KEY_PERTURB_FACTOR_S_LOCAL).asFloat();
    dst.perturbFactorSCTF = JSONCPP_READ_ERROR_HANDLER(src, "Professional", KEY_PERTURB_FACTOR_S_CTF).asFloat();
    dst.skipE = JSONCPP_READ_ERROR_HANDLER(src, "Professional", KEY_SKIP_E).asBool();
    dst.skipM = JSONCPP_READ_ERROR_HANDLER(src, "Professional", KEY_SKIP_M).asBool();
    dst.skipR = JSONCPP_READ_ERROR_HANDLER(src, "Professional", KEY_SKIP_R).asBool();
}

void logPara(const Json::Value src)
{
    Json::Value::Members mem = src.getMemberNames();

    for (size_t i = 0; i < mem.size(); i++)
    {
        if (src[mem[i]].type() == Json::objectValue)
        {
            logPara(src[mem[i]]);
        }

        else if (src[mem[i]].type() == Json::arrayValue)
        {
            for (int j = 0; j < (int)src[mem[i]].size(); j++)
            {
                logPara(src[mem[i]][j]);
            }
        }

        else if (src[mem[i]].type() == Json::stringValue)
        {
            CLOG(INFO, "LOGGER_SYS") << "[JSON PARAMTER] " << mem[i] << " : " << src[mem[i]].asString();
        }

        else if (src[mem[i]].type() == Json::realValue)
        {
            CLOG(INFO, "LOGGER_SYS") << "[JSON PARAMTER] " << mem[i] << " : " << src[mem[i]].asFloat();
        }

        else if (src[mem[i]].type() == Json::uintValue)
        {
            CLOG(INFO, "LOGGER_SYS") << "[JSON PARAMTER] " << mem[i] << " : " << src[mem[i]].asUInt();
        }

        else
        {
            CLOG(INFO, "LOGGER_SYS") << "[JSON PARAMTER] " << mem[i] << " : " << src[mem[i]].asInt();
        }
    }
}

INITIALIZE_EASYLOGGINGPP


void initGlobalPara(char *logFileFullName, Json::Reader &jsonReader, Json::Value &jsonRoot, OptimiserPara &thunderPara, const char *jsonFileName)
{
    ifstream jsonFile(jsonFileName, ios::binary);

    if (!jsonFile.is_open())
    {
        fprintf(stderr, "FAIL TO OPEN JSON [%s] PARAMETER FILE\n", jsonFileName);
        abort();
    }

    if (jsonReader.parse(jsonFile, jsonRoot))
    {
        readPara(thunderPara, jsonRoot);
    }
    else
    {
        fprintf(stderr, "THE FORMAT OF JSON FILE IS WRONG\n");
        abort();
    }

    jsonFile.close();
    char currWorkDir[FILE_NAME_LENGTH];
    memset(currWorkDir, '\0', sizeof(currWorkDir));
    GETCWD_ERROR_HANDLER(getcwd(currWorkDir, sizeof(currWorkDir)));
    char *outputDir = thunderPara.outputDirectory;

    if (strlen(outputDir) == 0)
    {
        strcpy(outputDir, "./");
    }

    size_t len = strlen(outputDir);

    /**
     *  Append '/' to the end of output if it is not end with '/'
     */
    if (outputDir[len - 1] != '/')
    {
        strcat(outputDir, "/");
    }

    char finalOutputDir[FILE_NAME_LENGTH];
    memset(finalOutputDir, '\0', sizeof(finalOutputDir));

    if (outputDir[0] != '/')
    {
        strcpy(finalOutputDir, currWorkDir);
        strcat(finalOutputDir, "/");

        /**
         *  Charactors "./" should not appear in the path
         */
        if (outputDir[0] == '.' && outputDir[1] == '/')
        {
            int k = 2;

            for (int i = strlen(finalOutputDir); outputDir[k] != '\0'; i ++)
            {
                finalOutputDir[i] = outputDir[k];
                k++;
            }
        }

        else
        {
            strcat(finalOutputDir, outputDir);
        }
    }

    else
    {
        strcpy(finalOutputDir, outputDir);
    }

    strcpy(logFileFullName, finalOutputDir);
    strcat(logFileFullName, "thunder.log");
    /**
     *  Construct value for dstPrefix
     */
    strcpy(thunderPara.dstPrefix, finalOutputDir);
    strcat(thunderPara.dstPrefix, thunderPara.outputFilePrefix);
    len = strlen(thunderPara.outputFilePrefix);

    if (len > 0 && thunderPara.outputFilePrefix[len - 1] != '_')
    {
        strcat(thunderPara.dstPrefix, "_");
    }

    strcpy(thunderPara.outputDirFullPath, finalOutputDir);
}

int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        cout << "Welcome to THUNDER "
             << THUNDER_VERSION_MAJOR
             << "."
             << THUNDER_VERSION_MINOR
             << "."
             << THUNDER_VERSION_ADDIT
             << "!"
             << endl
             << "Git Commit Version: "
             << COMMIT_VERSION_QUOTE
             << endl;
        return 0;
    }

    else if (argc != 2)
    {
        cout << "Wrong Number of Parameters Input!"
             << endl;
        return -1;
    }

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Json::Reader jsonReader;
    Json::Value jsonRoot;
    OptimiserPara thunderPara;
    char logFileFullName[FILE_NAME_LENGTH];
    memset(logFileFullName, '\0', sizeof(logFileFullName));
    /**
     *  Init some global parameters based on json file provided in argument argv[1]
     */
    initGlobalPara(logFileFullName, jsonReader, jsonRoot, thunderPara, argv[1]);
    initLogger(logFileFullName, rank);
    int flag = createCacheDirctory(thunderPara);
    if(flag == 1)
    {
        CLOG(INFO, "LOGGER_SYS") << "Cache directory is: " << thunderPara.cacheDirectory;
    }
    else
    {
        REPORT_ERROR("ERROR IN INITIALISING FFTW THREADS");
        abort();
    }

    if (rank == 0)
    {
        CLOG(INFO, "LOGGER_SYS") << "Git Commit Version: "
                                 << COMMIT_VERSION_QUOTE;
    }

    if (rank == 0)
    {
        CLOG(INFO, "LOGGER_SYS") << "THUNDER is Initiallised With "
                                 << size
                                 << " Processes";

        if (size <= 2)
        {
            CLOG(FATAL, "LOGGER_SYS") << "THUNDER REQUIRES AT LEAST 3 PROCESSES IN MPI";
            abort();
        }

        else if (size == 4)
        {
            CLOG(WARNING, "LOGGER_SYS") << "2 PROCESSES IN HEMISPHERE A, 1 PROCESS IN HEMISPHERE B, SEVERE INBALANCE";
        }
    }

    if (rank == 0) CLOG(INFO, "LOGGER_SYS") << "THUNDER v"
                                                << THUNDER_VERSION_MAJOR
                                                << "."
                                                << THUNDER_VERSION_MINOR
                                                << "."
                                                << THUNDER_VERSION_ADDIT;

#ifdef VERBOSE_LEVEL_1
    CLOG(INFO, "LOGGER_SYS") << "Initialising Processes";
#endif
#ifdef VERBOSE_LEVEL_1
    CLOG(INFO, "LOGGER_SYS") << "Process " << rank << " Initialised";
#endif

    if (rank == 0)
    {
        CLOG(INFO, "LOGGER_SYS") << "Logging JSON Parameters";
        logPara(jsonRoot);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        CLOG(INFO, "LOGGER_SYS") << "Setting Maximum Number of Threads Per Process";
    }

    omp_set_num_threads(thunderPara.nThreadsPerProcess);

    if (rank == 0)
    {
        CLOG(INFO, "LOGGER_SYS") << "Maximum Number of Threads in a Process is " << omp_get_max_threads();
    }

    if (rank == 0)
    {
        CLOG(INFO, "LOGGER_SYS") << "Initialising Threads Setting in FFTW";
    }

    if (TSFFTW_init_threads() == 0)
    {
        REPORT_ERROR("ERROR IN INITIALISING FFTW THREADS");
        abort();
    }

    if (rank == 0)
    {
        CLOG(INFO, "LOGGER_SYS") << "Setting Time Limit for Creating FFTW Plan";
    }

    TSFFTW_set_timelimit(60);

    if (rank == 0)
    {
        CLOG(INFO, "LOGGER_SYS") << "Setting Parameters";
    }

    Optimiser opt;
    opt.setPara(thunderPara);

    if (rank == 0)
    {
        CLOG(INFO, "LOGGER_SYS") << "Setting MPI Environment";
    }

    opt.setMPIEnv();

    if (rank == 0)
    {
        CLOG(INFO, "LOGGER_SYS") << "Running";
    }

    opt.run();
    MPI_Finalize();
    TSFFTW_cleanup_threads();
    return 0;
}
