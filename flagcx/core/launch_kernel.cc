#include "launch_kernel.h"

#include <dlfcn.h>
#include <stdio.h>

static launchAsyncKernel_t launchAsyncKernelFn = nullptr;

flagcxResult_t loadAsyncKernelSymbol() {
  void *handle = dlopen("devicefunc.so", RTLD_LAZY);
  if (!handle) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return flagcxRemoteError;
  }

  launchAsyncKernelFn = (launchAsyncKernel_t)dlsym(handle, "launchAsyncKernel");
  if (!launchAsyncKernelFn) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    return flagcxRemoteError;
  }

  return flagcxSuccess;
}

launchAsyncKernel_t getLaunchAsyncKernel() { return launchAsyncKernelFn; }
void cpuStreamWait(void *_args){
    bool * volatile args = (bool *) _args;
    __atomic_store_n(args, 1, __ATOMIC_RELAXED);
}


void cpuAsyncLaunch(void *_args){
    bool * volatile args = (bool *) _args;
    while(!__atomic_load_n(args, __ATOMIC_RELAXED));
    free(args);
}
