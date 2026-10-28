#include "./dpu_process/dpu_batch_kernel.h"


int main() 
{
    // 如果 DPU 程序被 host 调用多次，应该在开头使用 mem_reset()，以保证WRAM是干净的（放在末尾也可以）
    mem_reset(); 
    PushPackageHeader package_header;  // 读取 PushPackageHeader
    mram_read((__mram_ptr void *)recv_buffer, &package_header, sizeof(PushPackageHeader));

    if (package_header.option_type == 1) {
        if (me() == 0) {
            store_raw_vectors(&package_header);  // 保存数据的暂时是单线程
        }
        // 多线程保存数据实测几乎不会变快
    } else if (package_header.option_type == 2) {
        if (package_header.is_new == 1) {
            global_dpu_batch_id = 0;  // 重置查询的 batch id
            perfcounter_config(COUNT_CYCLES, true);  // 重置时间
        }
        process_query(&package_header);
    }
    
    return 0;
}

