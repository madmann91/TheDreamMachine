#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mem.h"

#if !defined(ALIGNED_MALLOC) && !defined(_MSC_VER)
void* aligned_malloc(int size, int align)
{
    unsigned char* mem = malloc(size + sizeof(void*) + align);
    size_t offset = (size_t)(mem + sizeof(void*)) % align;
    void** ptr = (void**)(mem + align + sizeof(void*) - offset);
    *(ptr - 1) = mem;
    return (void*)ptr;
}

void aligned_free(void* mem)
{
    void** ptr = (void**)mem;
    free(*(ptr - 1));
}
#endif

void buffer_add(mem_buffer* buf, void* data, unsigned int size)
{
    if (buf->size + size > buf->max_size)
    {
        buf->max_size = buf->max_size * 2 + size;
        buf->ptr = realloc(buf->ptr, buf->max_size);
    }

    memcpy(buf->ptr + buf->size, data, size);
    buf->size += size;
}

void buffer_refit(mem_buffer* buf)
{
    buf->ptr = realloc(buf->ptr, buf->size);
    buf->max_size = buf->size;
}
