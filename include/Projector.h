/*******************************************************************************
 * Authors Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef PROJECTOR_H
#define PROJECTOR_H

#include <armadillo>

#include "Complex.h"
#include "Error.h"
#include "Enum.h"

#include "Euler.h"

#include "Image.h"
#include "Volume.h"

#include "Coordinate5D.h"

using namespace arma;

class Projector
{
    private:

        int _maxRadius = -1;
        // Only the signal in the _maxRadius circle in Fourier space will be
        // processed. Otherwise, it will set to 0.
        // _maxRadius should be smaller than half of the shortest dimension
        // of the _projectee.
        // _maxRadius will be automatically set to the properly value when
        // projectee is set.

        Interpolation3DStyle _interpolation = linear3D;
        
        Volume _projectee;

    public:

        Projector();

        Projector(const Interpolation3DStyle interpolation);

        Projector(const Projector& that);

        ~Projector();

        Projector& operator=(const Projector& that);

        bool isEmpty() const;

        int maxRadius() const;

        void setMaxRadius(const int maxRadius); 

        Interpolation3DStyle interpolation() const;

        void setInterpolation(const Interpolation3DStyle interpolation);

        const Volume& projectee() const;

        void setProjectee(const Volume& src);

        void project(Image& dst,
                     const mat33& mat) const;

        void project(Image& dst,
                     const double phi,
                     const double theta,
                     const double psi) const;

        void project(Image& dst,
                     const double phi,
                     const double theta,
                     const double psi,
                     const float x,
                     const float y) const;

        void project(Image& dst,
                     const Coordinate5D& coordinate5D) const;

    private:

        void gridCorrection();
        /* perform gridding correction on _projectee */
};

#endif // PROJECTOR_H
