#include "heap.h"
#include <assert.h>
#include <stdio.h>
#include <stdbool.h>

void heap_insert(MaxHeap *heap, int index, uint32_t distance) 
{
    if (heap->size < MAX_RETURNED_PER_DPU) {
        heap->nodes[heap->size].index = index;
        heap->nodes[heap->size].distance = distance;
        heap->size++;

        // 上浮，保持最大堆结构
        uint32_t i = heap->size - 1;
        while (i > 0 && heap->nodes[(i - 1) / 2].distance < heap->nodes[i].distance) {
            HeapNode temp = heap->nodes[i];
            heap->nodes[i] = heap->nodes[(i - 1) / 2];
            heap->nodes[(i - 1) / 2] = temp;
            i = (i - 1) / 2;
        }
    } else if (distance < heap->nodes[0].distance) {
        // assert (heap->size == MAX_RETURNED_PER_DPU);
        // if (heap->nodes[0].index == 5460) {
        //     printf("node[%d] with distance:%u will evict nodes in heap!\n", index, distance);
        //     for (int j = 0; j < heap->size; j++) {
        //         printf("heap nodes[%d]:%d, distance:%u\n", j, heap->nodes[j].index, heap->nodes[j].distance);
        //     }
        // }
        // 新的更优，替换堆顶（当前最差），下沉重建堆
        heap->nodes[0].index = index;
        heap->nodes[0].distance = distance;

        // 下沉，保持最大堆结构
        uint32_t i = 0;
        while (2 * i + 1 < heap->size) {
            uint32_t largest = 2 * i + 1;

            // 找到子节点中距离最大的
            if (largest + 1 < heap->size &&
                heap->nodes[largest + 1].distance > heap->nodes[largest].distance) {
                largest++;
            }

            // 如果当前节点距离大于等于子节点距离，堆已经满足条件
            if (heap->nodes[i].distance >= heap->nodes[largest].distance) break;

            // 交换父子节点
            HeapNode temp = heap->nodes[i];
            heap->nodes[i] = heap->nodes[largest];
            heap->nodes[largest] = temp;

            // 更新i，继续下沉
            i = largest;
        }
        // 将堆顶距离更新到目前期望的最大距离
        thread_max_dist[me()] = heap->nodes[0].distance;
        // printf("thread_max_dist[%u]:%u\n", me(), thread_max_dist[me()]);
    }
}

// 判断给定 index 是否在堆中
bool is_in_heap(const MaxHeap *heap, int id)
{
    for (uint32_t i = 0; i < heap->size; ++i) {
        if (heap->nodes[i].index == id) {
            return true;
        }
    }
    return false;
}

