/** @file
 *  @author Mingxu Hu
 *  @version 1.4.14.190714
 *  @copyright GPLv2
 *
 *  ChangeLog
 *  AUTHOR     | TIME       | VERSION       | DESCRIPTION
 *  ------     | ----       | -------       | -----------
 *  Mingxu Hu  | 2019/07/14 | 1.4.14.190714 | new file
 *
 *  @brief 
 *
 */

#ifndef MEMORY_BAZAAR_CORE_UUID_H
#define MEMORY_BAZAAR_CORE_UUID_H

#include <string>
#include <unistd.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include "THUNDERConfig.h"
#include "Macro.h"
#include "Precision.h"
#include "Typedef.h"

inline std::string generateUUID()
{
    pid_t pid = getpid();

    char pidBuffer[32];
    sprintf(pidBuffer, "%ld", (long)pid);
    std::string uuid_string(pidBuffer);

    uuid_string.append("-");

    boost::uuids::uuid a_uuid = boost::uuids::random_generator()();
    
    uuid_string.append(boost::uuids::to_string(a_uuid));

    return uuid_string;
}

#endif // MEMORY_BAZAAR_CORE_UUID_H
