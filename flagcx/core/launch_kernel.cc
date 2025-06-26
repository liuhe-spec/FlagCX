#include "launch_kernel.h"

#include <dlfcn.h>
#include <stdio.h>

flagcxResult_t loadAsyncKernelSymbol(const char *path) {
  void *handle = dlopen(path, RTLD_LAZY);
  if (!handle) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return flagcxRemoteError;
  }

  deviceAdaptor->launchDeviceFunc = (launchAsyncKernel_t)dlsym(handle, "launchAsyncKernel");
  if (!deviceAdaptor->launchDeviceFunc) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    return flagcxRemoteError;
  }

  return flagcxSuccess;
}

launchAsyncKernel_t getLaunchAsyncKernel() {
  return deviceAdaptor->launchDeviceFunc;
}
void cpuStreamWait(void *_args){
    bool * volatile args = (bool *) _args;
    __atomic_store_n(args, 1, __ATOMIC_RELAXED);
}


void cpuAsyncLaunch(void *_args){
    bool * volatile args = (bool *) _args;
    while(!__atomic_load_n(args, __ATOMIC_RELAXED));
    free(args);
}
