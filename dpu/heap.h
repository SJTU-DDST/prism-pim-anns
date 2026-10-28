#ifndef HEAP_H
#define HEAP_H

#include <stdint.h>
#include <defs.h>  // 获取线程号函数 me()
#include "dpu_global.h"
#include "../share/global.h"


// 最大堆结构 - 维护 `MAX_RETURNED_PER_DPU` 个最近邻
typedef struct {
    int index;
    uint32_t distance;
} HeapNode;

// 这个堆只在 WRAM 中使用，因此不需要 8 字节对齐
typedef struct {
    HeapNode nodes[MAX_RETURNED_PER_DPU];
    uint32_t size;
} MaxHeap;

void heap_insert(MaxHeap *heap, int index, uint32_t distance);

bool is_in_heap(const MaxHeap *heap, int id);

#endif // HEAP_H
