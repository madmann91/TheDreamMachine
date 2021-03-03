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

#ifndef DREAM_MEM_H
#define DREAM_MEM_H

#include <stdint.h>

#if defined(_MSC_VER)
    #define SSE_ALIGN(var) __declspec(align(16)) var
    #define AVX_ALIGN(var) __declspec(align(32)) var
#elif defined(__GNUC__)
    #define SSE_ALIGN(var) var __attribute__ ((aligned (16)))
    #define AVX_ALIGN(var) var __attribute__ ((aligned (32)))
#else
    #error "no alignment directive found for your compiler"
#endif

#define IS_ALIGNED(ptr, align) (((uintptr_t)(const void *)(ptr)) % (align) == 0)

#if defined(_MSC_VER)
    /* MSVC has its own aligned malloc/free, so we use it */
    #include <malloc.h>
    #define aligned_malloc _aligned_malloc
    #define aligned_free _aligned_free
#else
    /* We provide our own implementation of the aligned malloc and
     * aligned free on non-MSVC compilers
     */
    void* aligned_malloc(int size, int align);
    void aligned_free(void* mem);
#endif

typedef struct
{
    unsigned int size;
    unsigned int max_size;
    unsigned char* ptr;
} mem_buffer;

/** Adds an element to a buffer, growing its size as needed */
void buffer_add(mem_buffer* buf, void* data, unsigned int size);
/** Refits a buffer to its exact size */
void buffer_refit(mem_buffer* buf);

#endif // DREAM_MEM_H
