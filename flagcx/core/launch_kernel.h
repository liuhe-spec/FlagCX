#ifndef FLAGCX_LAUNCH_KERNEL_H_
#define FLAGCX_LAUNCH_KERNEL_H_
#pragma once
#include "adaptor.h"
#include "debug.h"
#include "flagcx.h"
#include "param.h"
#include "topo.h"
#include "utils.h"
#include <dlfcn.h>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif


flagcxResult_t loadAsyncKernelSymbol(const char *path);

#ifdef __cplusplus
}
#endif

struct hostLaunchArgs{
    volatile bool stopLaunch;
    volatile bool retLaunch;
};

void cpuAsyncLaunch(void *_args);
void cpuStreamWait(void *_args);

#endif

