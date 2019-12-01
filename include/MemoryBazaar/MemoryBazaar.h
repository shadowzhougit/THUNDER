#include <iostream>
#include <omp.h>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <stdlib.h>
#include <stdio.h>

#include "THUNDERConfig.h"
#include "Macro.h"
#include "Precision.h"
#include "Typedef.h"
#include "Logging.h"

#include "core/UUID.h"

#ifndef MEMORY_BAZAAR_H
#define MEMORY_BAZAAR_H

enum BaseOrDerivedType { BaseType, DerivedType };

template<class T, BaseOrDerivedType bodt>
struct Assign {};

template<class T>
struct Assign<T, BaseType>
{
    inline static void read(T* a,
                            const int warehouseFD,
                            const size_t iItem,
                            const size_t nItem,
                            const size_t itemSize)
    {
        pread(warehouseFD, a, nItem * itemSize, iItem * itemSize);
    }

    inline static void write(const int warehouseFD,
                             const size_t iItem,
                             const size_t nItem,
                             const size_t itemSize,
                             const T* b)
    {
        pwrite(warehouseFD, b, nItem * itemSize, iItem * itemSize);
    }
};

template<class T>
struct Assign<T, DerivedType>
{
    inline static void read(T* a,
                            const int warehouseFD,
                            const size_t iItem,
                            const size_t nItem,
                            const size_t itemSize)
    {
        void* m = malloc(nItem * itemSize);

        pread(warehouseFD, m, nItem * itemSize, iItem * itemSize);

        for (size_t i = 0; i < nItem; i++)
        {
            deserialize(a[i], (void*)((char*)m + itemSize * i), itemSize);
        }

        free(m);
    } 

    inline static void write(const int warehouseFD,
                             const size_t iItem,
                             const size_t nItem,
                             const size_t itemSize,
                             const T* b)
    {
        void* m = malloc(nItem * itemSize);

        for (size_t i = 0; i < nItem; i++)
        {
            serialize((void*)((char*)m + itemSize * i), b[i]);
        }

        pwrite(warehouseFD, m, nItem * itemSize, iItem * itemSize);

        free(m);
    }
};

struct ReplaceRequest
{
    volatile long iPack;
    volatile unsigned int counter;

    ReplaceRequest(const size_t iiPack,
                   const unsigned int icounter)
    {
        iPack = iiPack;
        counter = icounter;
    }
};

template<class T>
struct Container
{
    Container()
    {
        iPack = -1;
        availability = false;
        referenceCounter = 0;
        item = NULL;
    };

    T* item;

    volatile long iPack;
    volatile bool availability;
    volatile int referenceCounter;
};

template<class T>
struct ThreadLastRecall
{
    long iLastVisitedPack;
    long iFirstItemLastVisitedPack;
    Container<T>* lastVisitedContainer;

    ThreadLastRecall()
    {
        iLastVisitedPack = -1;
        iFirstItemLastVisitedPack = INT_MAX;
        lastVisitedContainer = NULL;
    };
};

template<class T, BaseOrDerivedType bodt, size_t NUM_CONTAINER_PER_STALL = 4>
class MemoryBazaar
{
    private:

        std::string _UUID;

        unsigned int _maxThreads;

        size_t _nStall;

        /**
         * number of bytes an item will occupy
         */
        size_t _itemSize;

        size_t _packSize;

        ThreadLastRecall<T>* _threadLastRecall;

        long* _offsetToVisitNext;

        Container<T>* _bazaar;

        omp_lock_t* _mtx;

        int _warehouseFD;

        T* _category;

    public:

        MemoryBazaar() : _threadLastRecall(NULL),
                         _offsetToVisitNext(NULL),
                         _bazaar(NULL),
                         _mtx(NULL),
                         _category(NULL)
        {}

        MemoryBazaar(const unsigned int maxThreads,
                     const size_t nItem,
                     const size_t nStall,
                     const size_t itemSize,
                     const size_t packSize)
        {
            MemoryBazaar();

            setUp(maxThreads,
                  nItem,
                  nStall,
                  itemSize,
                  packSize);
        }

        inline unsigned int maxThreads() const { return _maxThreads; }
        
        inline size_t nStall() const { return _nStall; }

        inline size_t itemSize() const { return _itemSize; }

        inline size_t packSize() const { return _packSize; }

        void setUp(const unsigned int maxThreads,
                   const size_t nItem,
                   const size_t nStall,
                   const size_t itemSize,
                   const size_t packSize,
                   const char* cacheDirectory = NULL)
        {
            if (cacheDirectory == NULL)
            {
                _UUID = generateUUID();
            }
            else
            {
                _UUID = std::string(cacheDirectory).append(generateUUID());
            }

            _maxThreads = maxThreads;

            _nStall = nStall;

            _itemSize = itemSize;

            _packSize = packSize;


#ifdef APPLE
            _warehouseFD = open(_UUID.c_str(), O_RDWR | O_CREAT, 00644);
#else
            _warehouseFD = open(_UUID.c_str(), O_RDWR | O_CREAT | O_LARGEFILE, 00644);
#endif

            _threadLastRecall = new ThreadLastRecall<T>[_maxThreads];

            _offsetToVisitNext = new long[_maxThreads];

            for (size_t i = 0; i < _maxThreads; i++)
            {
                _offsetToVisitNext[i] = INT_MAX;
            }

            _bazaar = new Container<T>[_nStall * NUM_CONTAINER_PER_STALL];

            _mtx = new omp_lock_t[_nStall];

            for (size_t i = 0; i < _nStall; i++)
            {
                omp_init_lock(&_mtx[i]);
            }

            _category = new T[_nStall * NUM_CONTAINER_PER_STALL * _packSize];

            for (size_t i = 0; i < _nStall; i++)
            {
                for (size_t j = 0; j < NUM_CONTAINER_PER_STALL; j++)
                {
                    _bazaar[i * NUM_CONTAINER_PER_STALL + j].item = _category + i * NUM_CONTAINER_PER_STALL * _packSize + j * _packSize;
                }
            }
        }

        ~MemoryBazaar()
        {
            cleanUp();
        }

        void cleanUp()
        {
            if (_category != NULL)
            {
                delete[] _category;

                _category = NULL;
            }

            if (_mtx != NULL)
            {
                for (size_t i = 0; i < _nStall; i++)
                {
                    omp_destroy_lock(&_mtx[i]);
                }

                delete[] _mtx;

                _mtx = NULL;
            }

            if (_bazaar != NULL)
            {
                delete[] _bazaar;

                _bazaar = NULL;
            }

            if (_threadLastRecall != NULL)
            {
                delete[] _threadLastRecall;

                _threadLastRecall = NULL;
            }

            if (_offsetToVisitNext != NULL)
            {
                delete[] _offsetToVisitNext;

                _offsetToVisitNext = NULL;
            }

            remove(_UUID.c_str());
        }

        /**
         * whether it belongs to the same pack as the one previously visited
         */
        bool isSamePack(const size_t iItem) const
        {
            int threadID = omp_get_ancestor_thread_num(1);
            if (threadID == -1) threadID = 0;

            return isSamePack(iItem, threadID);
        }

        bool isSamePack(const size_t iItem,
                        const size_t threadID) const
        {
            long offset = iItem - _threadLastRecall[threadID].iFirstItemLastVisitedPack;

            if ((offset >= 0) && (offset < _packSize))
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        /**
         * record offset if the next vist to the same pack as the previous visit
         */
        bool isSamePackRegisterIItemToVisitNext(const size_t iItem)
        {
            int threadID = omp_get_ancestor_thread_num(1);
            if (threadID == -1) threadID = 0;

            return isSamePackRegisterIItemToVisitNext(iItem, threadID);
        }

        bool isSamePackRegisterIItemToVisitNext(const size_t iItem,
                                                const size_t threadID)
        {
            long offset = iItem - _threadLastRecall[threadID].iFirstItemLastVisitedPack;

            if ((offset >= 0) && (offset < _packSize))
            {
                _offsetToVisitNext[threadID] = offset;

                return true;
            }
            else
            {
                _offsetToVisitNext[threadID] = INT_MAX;

                return false;
            }
        }

        /**
         * currently, return the pack index
         * future, return the item reference
         */
        T& operator[](const size_t iItem)
        {
            int threadID = omp_get_ancestor_thread_num(1);
            if (threadID == -1) threadID = 0;

            if (_offsetToVisitNext[threadID] != INT_MAX)
            {
                long offset = _offsetToVisitNext[threadID];
                _offsetToVisitNext[threadID] = INT_MAX;

                // has been registed
                return _threadLastRecall[threadID].lastVisitedContainer->item[offset];
            }
            else if (isSamePackRegisterIItemToVisitNext(iItem, threadID))
            {
                long offset = _offsetToVisitNext[threadID];
                _offsetToVisitNext[threadID] = INT_MAX;

                // not been registed, but the same pack
                return _threadLastRecall[threadID].lastVisitedContainer->item[offset];
            }
            /***
            long offset = iItem - _threadLastRecall[threadID].iFirstItemLastVisitedPack;

            if ((offset >= 0) && (offset < _packSize))
            {
                return _threadLastRecall[threadID].lastVisitedContainer->item[offset];
            }
            ***/
            else
            {
                // not the same pack
                // minus 1 in referenceCounter of the last visited container
                if (_threadLastRecall[threadID].lastVisitedContainer != NULL)
                {
                    #pragma omp atomic
                    _threadLastRecall[threadID].lastVisitedContainer->referenceCounter -= 1;

                    _threadLastRecall[threadID].lastVisitedContainer = NULL;
                }

                // determine the pack index to be visited
                size_t iPack = iItem / _packSize;

                long offset = iItem - iPack * _packSize;

                // determine the stall to be visited
                size_t iStall = iPack % _nStall;

                // lock the iStall-th stall
                omp_set_lock(&_mtx[iStall]);

                // check the bazaar, find out whether the pack has already been loaded and it is available now
                for (size_t i = 0; i < NUM_CONTAINER_PER_STALL; i++)
                {
                    Container<T>* container = &_bazaar[iStall * NUM_CONTAINER_PER_STALL + i];

                    if ((container->iPack == iPack) &&
                        (container->availability))
                    {
                        #pragma omp atomic
                        container->referenceCounter += 1;

                        omp_unset_lock(&_mtx[iStall]);

                        _threadLastRecall[threadID].iLastVisitedPack = iPack;
                        _threadLastRecall[threadID].iFirstItemLastVisitedPack= iPack * _packSize;
                        _threadLastRecall[threadID].lastVisitedContainer = &_bazaar[iStall * NUM_CONTAINER_PER_STALL + i];

                        return container->item[offset];
                    }
                }

                ReplaceRequest request(iPack, 1);

                for (size_t i = 0; i < NUM_CONTAINER_PER_STALL; i++)
                {
                    Container<T>* container = &_bazaar[iStall * NUM_CONTAINER_PER_STALL + i];

                    if (container->iPack == -1)
                    {
                        container->iPack = request.iPack;

                        container->referenceCounter = request.counter;
                        container->availability = true;

                        omp_unset_lock(&_mtx[iStall]);

                        _threadLastRecall[threadID].iLastVisitedPack = iPack;
                        _threadLastRecall[threadID].iFirstItemLastVisitedPack= iPack * _packSize;
                        _threadLastRecall[threadID].lastVisitedContainer = container; 

                        return container->item[offset];
                    }
                }

                Container<T>* container = &_bazaar[(iStall + 1) * NUM_CONTAINER_PER_STALL - 1];

                // make the container which to be replaced unavailable
                container->availability = false;

                while ((request.iPack != iPack) ||
                       (container->referenceCounter > 0))
                {
                    /***
                    std::cout << "thread " << threadID << "is waiting for other threads ending its life cycle" << std::endl;
                    size_t theOtherThreadID;
                    if (threadID == 0)
                    {
                        theOtherThreadID = 1;
                    }
                    else
                    {
                        theOtherThreadID = 0;
                    }

                    if (_threadLastRecall[theOtherThreadID].lastVisitedContainer != NULL)
                    {
                        std::cout << "_threadLastRecall[theOtherThreadID].lastVisitedContainer = " << _threadLastRecall[theOtherThreadID].lastVisitedContainer << std::endl;
                    }
                    else
                    {
                        std::cout << "_threadLastRecall[theOtherThreadID == NULL" << std::endl;
                    }
                    ***/
                }

                // now, it is time
                // REPLACE OPERATION

                if (container->iPack != request.iPack)
                {
                    Assign<T, bodt>::write(_warehouseFD, container->iPack * _packSize, _packSize, _itemSize, container->item);
                    Assign<T, bodt>::read(container->item, _warehouseFD, iPack * _packSize, _packSize, _itemSize);
                }

                // renew information

                container->referenceCounter = request.counter;

                container->iPack = iPack;

                container->availability = true;

                omp_unset_lock(&_mtx[iStall]);

                _threadLastRecall[threadID].iLastVisitedPack = iPack;
                _threadLastRecall[threadID].iFirstItemLastVisitedPack= iPack * _packSize;
                _threadLastRecall[threadID].lastVisitedContainer = container;

                return container->item[offset];
            }
        }

        void cleanThread(const int threadID)
        {
            if (_threadLastRecall != NULL)
            {
                _threadLastRecall[threadID].iLastVisitedPack = -1;
                _threadLastRecall[threadID].iFirstItemLastVisitedPack = INT_MAX;

                if (_threadLastRecall[threadID].lastVisitedContainer != NULL)
                {
                    #pragma omp atomic
                    _threadLastRecall[threadID].lastVisitedContainer->referenceCounter -= 1;

                    _threadLastRecall[threadID].lastVisitedContainer = NULL;
                }
            }
        }

        void endLastVisit()
        {
            int threadID = omp_get_ancestor_thread_num(1);
            if (threadID == -1) threadID = 0;

            cleanThread(threadID);
        }

        void endLastVisit(const size_t iItemIntendToVisitNext)
        {
            if (!isSamePackRegisterIItemToVisitNext(iItemIntendToVisitNext))
            {
                endLastVisit();
            }
        }
};

template<class T, BaseOrDerivedType bodt, size_t NUM_CONTAINER_PER_STALL = 4>
class MemoryBazaarDustman
{
    public:

    MemoryBazaar<T, bodt, NUM_CONTAINER_PER_STALL>* _target;

    MemoryBazaarDustman() : _target(NULL) {}

    MemoryBazaarDustman(MemoryBazaar<T, bodt, NUM_CONTAINER_PER_STALL>* target)
    {
        _target = target;
    }

    MemoryBazaarDustman(const MemoryBazaarDustman<T, bodt, NUM_CONTAINER_PER_STALL>& that)
    {
        _target = that._target;
    }

    ~MemoryBazaarDustman()
    {
        // int threadID = omp_get_thread_num();
        int threadID = omp_get_ancestor_thread_num(1);

        if (threadID == -1) threadID = 0;
        
        if (_target != NULL)
        {
            _target->cleanThread(threadID);

            _target = NULL;
        }
    }
};

#endif // MEMORY_BAZAAR_H
