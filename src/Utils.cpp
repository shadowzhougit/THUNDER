#include "Utils.h"

#include <stdexcept>
#include <regex.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>

namespace {
class Regex {
private:
    regex_t regex;

public:
    Regex(const char* pattern, int flags)
    {
        int rc = regcomp(&regex, pattern, flags);
        if (rc != 0) {
            char buf[3000];
            regerror(rc, NULL, buf, sizeof(buf));
            throw std::invalid_argument(buf);
        }
    }

    regex_t* getInternal()
    {
        return &regex;
    }

    ~Regex()
    {
        regfree(&regex);
    }
};
}

bool regexMatches(const char* str, const char* pattern)
{
    Regex regex(pattern, REG_EXTENDED | REG_NOSUB);
    int rc = regexec(regex.getInternal(), str, 0, NULL, 0);
    if (rc == 0)
        return true;
    if (rc == REG_NOMATCH)
        return false;

    char buf[3000];
    regerror(rc, regex.getInternal(), buf, sizeof(buf));
    throw std::runtime_error(buf);
}

const char* getTempDirectory(void)
{
    static const char* tmp = NULL;
    if (tmp)
        return tmp;
    if (access("/tmp", R_OK | W_OK | X_OK) == 0) {
        tmp = "/tmp";
        return tmp;
    }
    tmp = "./tmp";
    mkdir(tmp, 0755);
    return tmp;
}

void optionCheck(char option[], int size, const struct option long_options[])
{
    int len = 0;
    int i = 0;
    int existFlag = 0;

    while (long_options[len].name != NULL)
    {
        len++;
    }

    while (i < size)
    {
        if (option[i] != '\0')
        {
            int flag = 0;

            for (int j = 0; j < len; j++)
            {
                if (option[i] == long_options[j].val)
                {
                    cout << "Option --" << long_options[j].name << " does not exist." << endl;
                    flag = 1;
                    existFlag += 1;
                    break;
                }
            }
            if (flag == 0)
            {
                cout << "Option -" << option[i] << " does not exist." << endl;
                existFlag += 1;
            }
        }
        i++;
    }

    if (existFlag)
    {
        exit(1);
    }
}
