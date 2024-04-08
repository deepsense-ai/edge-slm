#include <sys/resource.h>
namespace ds {

double get_peak_process_mem_usage_gb() {
    rusage mem_usage;
    getrusage(RUSAGE_SELF, &mem_usage);

    auto peak_gb = mem_usage.ru_maxrss / 1024.0 / 1024.0;
    return peak_gb;
}

}