/** (C) 2013-2014 MadMann's Company
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#ifndef DREAM_LINALG_INL
#define DREAM_LINALG_INL

#include <math.h>

/* These functions are provided for convenience only. They should not
 * be used in production code, since an implementation using SSE intrinsics
 * will generally be faster.
 */

typedef float vector3[3];
typedef float vector4[4];
typedef float quat[4];
typedef float matrix3[9];
typedef float matrix4[16];

static inline void vector3_copy(const vector3 a, vector3 b)
{
    b[0] = a[0];
    b[1] = a[1];
    b[2] = a[2];
}

static inline void vector3_set(float a, float b, float c, vector3 d)
{
    d[0] = a;
    d[1] = b;
    d[2] = c;
}

static inline void vector3_set_all(float a, vector3 b)
{
    b[0] = a;
    b[1] = a;
    b[2] = a;
}

static inline void vector3_negate(const vector3 a, vector3 b)
{
    b[0] = -a[0];
    b[1] = -a[1];
    b[2] = -a[2];
}

static inline void vector3_add(const vector3 a, const vector3 b, vector3 c)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
    c[2] = a[2] + b[2];
}

static inline void vector3_subtract(const vector3 a, const vector3 b, vector3 c)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
}

static inline void vector3_max(const vector3 a, const vector3 b, vector3 c)
{
    c[0] = (a[0] > b[0]) ? a[0] : b[0];
    c[1] = (a[1] > b[1]) ? a[1] : b[1];
    c[2] = (a[2] > b[2]) ? a[2] : b[2];
}

static inline void vector3_min(const vector3 a, const vector3 b, vector3 c)
{
    c[0] = (a[0] < b[0]) ? a[0] : b[0];
    c[1] = (a[1] < b[1]) ? a[1] : b[1];
    c[2] = (a[2] < b[2]) ? a[2] : b[2];
}

static inline void vector3_scale(const vector3 a, float b, vector3 c)
{
    c[0] = a[0] * b;
    c[1] = a[1] * b;
    c[2] = a[2] * b;
}

static inline void vector3_scale3(const vector3 a, const vector3 b, vector3 c)
{
    c[0] = a[0] * b[0];
    c[1] = a[1] * b[1];
    c[2] = a[2] * b[2];
}

static inline void vector3_cross(const vector3 a, const vector3 b, vector3 c)
{
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

static inline float vector3_dot(const vector3 a, const vector3 b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline float vector3_length2(const vector3 a)
{
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}

static inline float vector3_length(const vector3 a)
{
    return sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

static inline void vector4_copy(const vector4 a, vector4 b)
{
    b[0] = a[0];
    b[1] = a[1];
    b[2] = a[2];
    b[3] = a[3];
}

static inline void vector4_add(const vector4 a, const vector4 b, vector4 c)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
    c[2] = a[2] + b[2];
    c[3] = a[3] + b[3];
}

static inline void vector4_max(const vector4 a, const vector4 b, vector4 c)
{
    c[0] = (a[0] > b[0]) ? a[0] : b[0];
    c[1] = (a[1] > b[1]) ? a[1] : b[1];
    c[2] = (a[2] > b[2]) ? a[2] : b[2];
    c[3] = (a[3] > b[3]) ? a[3] : b[3];
}

static inline void vector4_min(const vector4 a, const vector4 b, vector4 c)
{
    c[0] = (a[0] < b[0]) ? a[0] : b[0];
    c[1] = (a[1] < b[1]) ? a[1] : b[1];
    c[2] = (a[2] < b[2]) ? a[2] : b[2];
    c[3] = (a[3] < b[3]) ? a[3] : b[3];
}

static inline void vector4_scale(const vector4 a, float b, vector4 c)
{
    c[0] = a[0] * b;
    c[1] = a[1] * b;
    c[2] = a[2] * b;
    c[3] = a[3] * b;
}

static inline void vector4_scale4(const vector4 a, const vector4 b, vector4 c)
{
    c[0] = a[0] * b[0];
    c[1] = a[1] * b[1];
    c[2] = a[2] * b[2];
    c[3] = a[3] * b[3];
}

static inline void quat_mult(const quat a, const quat b, quat c)
{
    c[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
    c[1] = a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0];
    c[2] = a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3];
    c[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
}

static inline void quat_conj(const quat a, quat b)
{
    b[0] = -a[0];
    b[1] = -a[1];
    b[2] = -a[2];
    b[3] = a[3];
}

#endif // DREAM_LINALG_INL
