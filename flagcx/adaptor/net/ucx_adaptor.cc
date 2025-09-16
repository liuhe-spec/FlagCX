#ifdef USE_UCX

/*************************************************************************
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "adaptor.h"
#include "comm.h"
#include "core.h"
#include "net.h"
#include "param.h"
#include "socket.h"
#include "utils.h"
#include "flagcx.h"
#include "debug.h"
#include "ibvwrap.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <uct/api/uct.h>



// UCX UCT constants
#define FLAGCX_UCX_UCT_MAX_RECVS 8
#define FLAGCX_UCT_AM_RTR 14
#define FLAGCX_UCT_AM_ATP 15
#define FLAGCX_UCT_LISTEN_HANDLE_MAGIC 0x43cf19ed91abdb85
#define FLAGCX_UCT_REG_ALIGN 4096
#define MAX_IB_DEVS 32
#define MAXNAMESIZE 64
#define PATH_MAX 4096
#define FLAGCX_NET_MAX_DEVS_PER_NIC 4
#define FLAGCX_NET_IB_MAX_RECVS 8


// SOCKET_NAME_MAXLEN already defined in socket.h

// UCXCHECK macro
#define UCXCHECK(statement, failure_action, message, ...) \
  do { \
    ucs_status_t _status = statement; \
    if (_status != UCS_OK) { \
      WARN("Failed: " message ": %s", ##__VA_ARGS__, \
           ucs_status_string(_status)); \
      failure_action; \
    } \
  } while (0)

#define FLAGCX_UCT_START 0
#define FLAGCX_UCT_CONNECT 1
#define FLAGCX_UCT_RECEIVE_ADDR 2
#define FLAGCX_UCT_ACCEPT 3
#define FLAGCX_UCT_RECEIVE_REMOTE 4
#define FLAGCX_UCT_RX_READY 5
#define FLAGCX_UCT_DONE 6



#define FLAGCX_IB_LLSTR(x) ((x) == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE")

// Missing enum
enum flagcxIbProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
  IB_PROVIDER_MLX4 = 2
};

// Missing string array
static const char* ibProviderName[] = {
  "NONE",
  "MLX5", 
  "MLX4"
};

#define FLAGCX_IB_MAX_DEVS_PER_NIC 4
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE*FLAGCX_IB_MAX_DEVS_PER_NIC)+FLAGCX_IB_MAX_DEVS_PER_NIC
typedef struct {
  int ndevs;
  int devs[FLAGCX_NET_MAX_DEVS_PER_NIC]; // FLAGCX_IB_MAX_DEVS_PER_NIC
} flagcxNetVDeviceProps_t;
typedef struct flagcxIbMergedDev {
  flagcxNetVDeviceProps_t vProps;
  int speed;
  char devName[MAX_MERGED_DEV_NAME]; 
} __attribute__((aligned(64))) flagcxIbMergedDev;

#define MAX_IB_DEVS  32
#define MAX_IB_VDEVS MAX_IB_DEVS*8
extern struct flagcxIbMergedDev flagcxIbMergedDevs[MAX_IB_VDEVS];
extern struct flagcxIbDev flagcxIbDevs[MAX_IB_DEVS];
extern int flagcxNIbDevs;
static int flagcxNMergedIbDevs = -1;
static pthread_mutex_t flagcx_uct_lock = PTHREAD_MUTEX_INITIALIZER;
extern pthread_t flagcxIbAsyncThread;

// Forward declarations and missing definitions
// UCX type definitions
typedef struct {
  uint8_t dev_addr_size;
  uint8_t ep_addr_size;
  uint8_t data[64];
} flagcx_uct_ep_addr_t;


typedef struct {
  uct_iface_h     iface;
  uct_md_h        md;
  uct_component_h comp;
  void            *addr;
  size_t          addr_size;
  void            *dev_addr;
  size_t          dev_addr_size;
  size_t          ep_addr_size;
  size_t          rkey_packed_size;

  size_t          am_max_short;
  size_t          min_get_zcopy;
} flagcx_uct_iface_t;



typedef struct {
  uct_ep_h         ep;
  uct_ep_addr_t    *addr;
  size_t           addr_size;
  flagcx_uct_iface_t *uct_iface;
  uint8_t          data[];
} flagcx_uct_ep_t;

typedef struct flagcx_uct_worker {
  struct flagcx_uct_worker *next;
  struct {
    pthread_t thread;
    int       dev;
  } id;

  int                     count;
  ucs_async_context_t     *async;
  uct_worker_h            worker;
  flagcx_uct_iface_t        *uct_iface;
  struct flagcx_uct_context *context;
} flagcx_uct_worker_t;


/* All the remote addresses for the communicator */
typedef struct flagcx_uct_comm_addr {
  flagcx_uct_ep_addr_t rma;
  /* TODO: Add multi-QP here */
} flagcx_uct_comm_addr_t;

typedef struct flagcx_uct_comm {
  struct flagcxSocket       sock;
  struct flagcx_uct_context *context;
  int                     dev;

  flagcx_uct_worker_t       *uct_worker;
  flagcx_uct_iface_t        *uct_iface;
  flagcx_uct_ep_t           *uct_ep;

  struct {
    flagcx_uct_comm_addr_t       addr;  /* Remote addresses */
    const struct flagcx_uct_comm *comm; /* Cookie received in connect */
  } remote;

  /* Local GET on current device */
  struct {
    int                enabled;
    flagcx_uct_ep_t      *uct_ep; /* Locally read from HCA */
    flagcx_uct_ep_addr_t addr;

    uint8_t            *mem; /* Dummy memory to read into */
    uct_mem_h          memh;
  } gpu_flush;
} flagcx_uct_comm_t;

/* Global state of the plugin */
typedef struct flagcx_uct_context {
  /* Transport to use */
  const char              *tl_name;

  /* IB devices available */
  int                     dev_count;
  int                     merge_dev_count;

  /* Use by common code to setup communicators */
  struct flagcx_uct_ops {
    flagcxResult_t (*comm_alloc)(flagcx_uct_comm_t **comm);
    flagcxResult_t (*comm_init)(flagcx_uct_comm_t *comm,
                              struct flagcx_uct_context *context,
                              flagcx_uct_worker_t *worker, int dev,
                              const flagcx_uct_comm_t *remote_comm);
    flagcxResult_t (*iface_set)(flagcx_uct_iface_t *uct_iface);
  } ops;

  /* Max sizes needed */
  size_t                  am_short_size;
  size_t                  rkey_size;

  /* OOB socket for accepting/connecting */
  char                    if_name[MAX_IF_NAME_SIZE];
  union flagcxSocketAddress if_addr;

  /* Number of listener created */
  uint32_t                listener_count;

  /* List of created workers */
  flagcx_uct_worker_t       *worker_list;
} flagcx_uct_context_t;

typedef struct flagcxIbStats {
  int fatalErrorCount;
} flagcxIbStats;

struct flagcxIbMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  ibv_mr *mr;
};

struct flagcxIbMrCache {
  struct flagcxIbMr *slots;
  int capacity, population;
};

 typedef struct flagcxIbDev {
   pthread_mutex_t lock;
   int      device;
   uint64_t guid;
   uint8_t portNum;
   uint8_t  link;
   uint8_t  isSharpDev;
   int      speed;
   struct   ibv_context* context;
   int      pdRefs;
   struct ibv_pd*  pd;
   char     devName[MAXNAMESIZE];
   char     *pciPath;
   char* virtualPciPath;
   int      realPort;
   int      maxQp;
   float latency;
   struct   flagcxIbMrCache mrCache;
   int ar; // ADAPTIVE_ROUTING
   struct ibv_port_attr portAttr;
   struct flagcxIbStats stats;
   int dmaBufSupported;
   enum flagcxIbProvider ibProvider;
   union {
  struct {
       int dataDirect;
     } mlx5;
   } capsProvider;
 } __attribute__((aligned(64))) flagcxIbDev;

// Global variables will be defined after structures

// Parameter declarations (defined in ibrc_adaptor.cc)
extern int flagcxParamIbMergeVfs(void);
extern int flagcxParamIbAdaptiveRouting(void);
extern int flagcxParamIbMergeNics(void);
FLAGCX_PARAM(SharpMaxComms, "SHARP_MAX_COMMS", 1);





// Missing arrays and constants
static int ibv_widths[] = {1, 2, 4, 8, 12};
static int ibv_speeds[] = {2500, 5000, 10000, 14000, 25000, 50000, 100000, 200000};

// Helper function
static int first_bit_set(int value, int max_bits) {
  for (int i = 0; i <= max_bits; i++) {
    if (value & (1 << i)) return i;
  }
  return 0;
}
// Global variables defined below
// Missing structure definitions



// Missing variables
// ibDmaSupportInitDev defined later as thread-local
static struct {
  pthread_once_t once;
} onces[MAX_IB_DEVS];



static int flagcxIbMatchVfPath(char* path1, char* path2) {
  // Merge multi-port NICs into the same PCI device
  if (flagcxParamIbMergeVfs()) {
    return strncmp(path1, path2, strlen(path1)-4) == 0;
  } else {
    return strncmp(path1, path2, strlen(path1)-1) == 0;
  }
}
 static void flagcx_uct_ep_destroy(flagcx_uct_ep_t *uct_ep) {
   uct_ep_destroy(uct_ep->ep);
   free(uct_ep);
 }

flagcxResult_t flagcxIbStatsInit(struct flagcxIbStats* stat) {
  __atomic_store_n(&stat->fatalErrorCount, 0, __ATOMIC_RELAXED);
  return flagcxSuccess;
}

flagcxResult_t flagcx_p2p_ib_pci_path(flagcxIbDev *devs, int num_devs, char* dev_name, char** path, int* real_port)
{
  char device_path[PATH_MAX];
  snprintf(device_path, PATH_MAX, "/sys/class/infiniband/%s/device", dev_name);
  char* p = realpath(device_path, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s", device_path);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p)-1] = '0';
    // Also merge virtual functions (VF) into the same device
    if (flagcxParamIbMergeVfs()) p[strlen(p)-3] = p[strlen(p)-4] = '0';
    // Keep the real port aside (the ibv port is always 1 on recent cards)
    *real_port = 0;
    for (int d=0; d<num_devs; d++) {
      if (flagcxIbMatchVfPath(p, flagcxIbDevs[d].pciPath)) (*real_port)++;
    }
  }
  *path = p;
  return flagcxSuccess;
}

static const char *flagcx_dev_name(int dev) {
  static __thread char buf[128];
  snprintf(buf, sizeof(buf), "%s:%d", flagcxIbDevs[dev].devName,
           flagcxIbDevs[dev].portNum);
  return buf;
}

int flagcx_p2p_ib_width(int width)
{
  return ibv_widths[first_bit_set(width, sizeof(ibv_widths)/sizeof(int)-1)];
}

int flagcx_p2p_ib_speed(int speed)
{
  return ibv_speeds[first_bit_set(speed, sizeof(ibv_speeds)/sizeof(int)-1)];
}

static __thread int ibDmaSupportInitDev; // which device to init, must be thread local
static void ibDmaBufSupportInitOnce(){
  flagcxResult_t res;
  int dev_fail = 0;

  // This is a physical device, not a virtual one, so select from ibDevs
  flagcxIbMergedDev* mergedDev = flagcxIbMergedDevs + ibDmaSupportInitDev;
  flagcxIbDev* ibDev = flagcxIbDevs + mergedDev->vProps.devs[0];
  struct ibv_pd* pd;
  struct ibv_context* ctx = ibDev->context;
  FLAGCXCHECKGOTO(flagcxWrapIbvAllocPd(&pd, ctx), res, failure);
  // Test kernel DMA-BUF support with a dummy call (fd=-1)
  (void)flagcxWrapDirectIbvRegMr(pd, 0ULL /*addr*/, 0ULL /*len*/, 0 /*access*/);
  // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  FLAGCXCHECKGOTO(flagcxWrapIbvDeallocPd(pd), res, failure);
  // stop the search and goto failure
  if (dev_fail) goto failure;
  ibDev->dmaBufSupported = 1;
  return;
failure:
  ibDev->dmaBufSupported = -1;
  return;
}
#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)
static int flagcxIbGdrModuleLoaded = 0; // 1 = true, 0 = false

static void ibGdrSupportInitOnce() {
  // Check for the nv_peer_mem module being loaded
  flagcxIbGdrModuleLoaded = KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem/version") ||
                          KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem_nc/version") ||
                          KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
}

static void flagcxIbStatsFatalError(struct flagcxIbStats* stat){
  __atomic_fetch_add(&stat->fatalErrorCount, 1, __ATOMIC_RELAXED);
}
static void flagcxIbQpFatalError(struct ibv_qp* qp) {
  flagcxIbStatsFatalError((struct flagcxIbStats*)qp->qp_context);
}
static void flagcxIbCqFatalError(struct ibv_cq* cq) {
  flagcxIbStatsFatalError((struct flagcxIbStats*)cq->cq_context);
}
static void flagcxIbDevFatalError(struct flagcxIbDev* dev) {
  flagcxIbStatsFatalError(&dev->stats);
}

static void* flagcxIbAsyncThreadMain(void* args) {
  struct flagcxIbDev* dev = (struct flagcxIbDev*)args;
  while (1) {
    struct ibv_async_event event;
    if (flagcxSuccess != flagcxWrapIbvGetAsyncEvent(dev->context, &event)) { break; }
    char *str;
    struct ibv_cq* cq = event.element.cq;    // only valid if CQ error
    struct ibv_qp* qp = event.element.qp;    // only valid if QP error
    struct ibv_srq* srq = event.element.srq; // only valid if SRQ error
    if (flagcxSuccess != flagcxWrapIbvEventTypeStr(&str, event.event_type)) { break; }
    switch (event.event_type) {
    case IBV_EVENT_DEVICE_FATAL:
      // the above is device fatal error
      WARN("NET/IB : %s:%d async fatal event: %s", dev->devName, dev->portNum, str);
      flagcxIbDevFatalError(dev);
      break;
    case IBV_EVENT_CQ_ERR:
      // the above is a CQ fatal error
      WARN("NET/IB : %s:%d async fatal event on CQ (%p): %s", dev->devName, dev->portNum, cq, str);
      flagcxIbCqFatalError(cq);
      break;
    case IBV_EVENT_QP_FATAL:
    case IBV_EVENT_QP_REQ_ERR:
    case IBV_EVENT_QP_ACCESS_ERR:
      // the above are QP fatal errors
      WARN("NET/IB : %s:%d async fatal event on QP (%p): %s", dev->devName, dev->portNum, qp, str);
      flagcxIbQpFatalError(qp);
      break;
    case IBV_EVENT_SRQ_ERR:
      // SRQ are not used in flagcx
      WARN("NET/IB : %s:%d async fatal event on SRQ, unused for now (%p): %s", dev->devName, dev->portNum, srq, str);
      break;
    case IBV_EVENT_PATH_MIG_ERR:
    case IBV_EVENT_PORT_ERR:
    case IBV_EVENT_PATH_MIG:
    case IBV_EVENT_PORT_ACTIVE:
    case IBV_EVENT_SQ_DRAINED:
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_QP_LAST_WQE_REACHED:
    case IBV_EVENT_CLIENT_REREGISTER:
    case IBV_EVENT_SRQ_LIMIT_REACHED:
      // the above are non-fatal
      WARN("NET/IB : %s:%d Got async error event: %s", dev->devName, dev->portNum, str);
      break;
    case IBV_EVENT_COMM_EST:
      break;
    default:
      WARN("NET/IB : %s:%d unknown event type (%d)", dev->devName, dev->portNum, event.event_type);
      break;
    }
    // acknowledgment needs to happen last to avoid user-after-free
    if (flagcxSuccess != flagcxWrapIbvAckAsyncEvent(&event)) { break; }
  }
  return NULL;
}

flagcxResult_t flagcx_p2p_dmabuf_support(int dev) {
  // init the device only once
  ibDmaSupportInitDev = dev;
  pthread_once(&onces[dev].once, ibDmaBufSupportInitOnce);
  flagcxIbMergedDev* mergedDev = flagcxIbMergedDevs + ibDmaSupportInitDev;
  flagcxIbDev* ibDev = flagcxIbDevs + mergedDev->vProps.devs[0];
  int dmaBufSupported = ibDev->dmaBufSupported;
  if (dmaBufSupported == 1) return flagcxSuccess;
  return flagcxSystemError;
}

flagcxResult_t flagcxIbMakeVDeviceInternal(int* d, flagcxNetVDeviceProps_t* props, int flagcxNIbDevs, int *flagcxNMergedIbDevs) {
  if ((flagcxParamIbMergeNics() == 0) && props->ndevs > 1) {
    INFO(FLAGCX_NET, "NET/IB : Skipping makeVDevice, flagcx_IB_MERGE_NICS=0");
    return flagcxInvalidUsage;
  }

  if (props->ndevs == 0) {
   WARN("NET/IB : Can't make virtual NIC with 0 devices");
   return flagcxInvalidUsage;
  }

  if (*flagcxNMergedIbDevs == MAX_IB_DEVS) {
    WARN("NET/IB : Cannot allocate any more virtual devices (%d)", MAX_IB_DEVS);
    return flagcxInvalidUsage;
  }

  // Always count up number of merged devices
  flagcxIbMergedDev* mDev = flagcxIbMergedDevs + *flagcxNMergedIbDevs;
  mDev->vProps.ndevs = 0;
  mDev->speed = 0;

  for (int i = 0; i < props->ndevs; i++) {
    flagcxIbDev* dev = flagcxIbDevs + props->devs[i];
    if (mDev->vProps.ndevs == 2) return flagcxInvalidUsage; // FLAGCX_IB_MAX_DEVS_PER_NIC
    mDev->vProps.devs[mDev->vProps.ndevs++] = props->devs[i];
    mDev->speed += dev->speed;
    // Each successive time, copy the name '+' new name
    if (mDev->vProps.ndevs > 1) {
      snprintf(mDev->devName + strlen(mDev->devName), sizeof(mDev->devName) - strlen(mDev->devName), "+%s", dev->devName);
    // First time, copy the plain name
    } else {
      strncpy(mDev->devName, dev->devName, MAXNAMESIZE);
     }
   }

  // Check link layers
  flagcxIbDev* dev0 = flagcxIbDevs + props->devs[0];
  for (int i = 1; i < props->ndevs; i++) {
    if (props->devs[i] >= flagcxNIbDevs) {
      WARN("NET/IB : Cannot use physical device %d, max %d", props->devs[i], flagcxNIbDevs);
      return flagcxInvalidUsage;
    }
    flagcxIbDev* dev = flagcxIbDevs + props->devs[i];
    if (dev->link != dev0->link) {
      WARN("NET/IB : Attempted to merge incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using flagcx_IB_HCA",
        props->devs[0], dev0->devName, dev0->portNum, "IB", props->devs[i], dev->devName, dev->portNum, "IB"); 
      return flagcxInvalidUsage;
    }
  }

  *d = *flagcxNMergedIbDevs;
  (*flagcxNMergedIbDevs)++;

  INFO(FLAGCX_NET, "NET/IB : Made virtual device [%d] name=%s speed=%d ndevs=%d", *d, mDev->devName, mDev->speed, mDev->vProps.ndevs);
  return flagcxSuccess;
}

 static flagcx_uct_ep_t *flagcx_uct_ep_create(flagcx_uct_iface_t *uct_iface) {
   flagcx_uct_ep_t *uct_ep;
   uct_ep_params_t ep_params;
 
       uct_ep = (flagcx_uct_ep_t*)calloc(1, sizeof(*uct_ep) + uct_iface->ep_addr_size);
   if (uct_ep == NULL) {
     WARN("Failed to alloc EP memory");
     return NULL;
   }
 
   uct_ep->addr      = (uct_ep_addr_t*)uct_ep->data;
   uct_ep->uct_iface = uct_iface;
 
   ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
   ep_params.iface      = uct_iface->iface;
 
   UCXCHECK(uct_ep_create(&ep_params, &uct_ep->ep), goto fail, "create UCT EP");
   UCXCHECK(uct_ep_get_address(uct_ep->ep, uct_ep->addr), goto fail_destroy,
            "get UCT EP address");
 
   return uct_ep;
 
 fail_destroy:
   flagcx_uct_ep_destroy(uct_ep);
 fail:
   free(uct_ep);
   return NULL;
 }

 static uct_iface_h flagcx_uct_resource_iface_open(uct_worker_h worker,
                                                 uct_md_h md,
                                                 uct_tl_resource_desc_t *tl) {
   uct_iface_params_t params = {};
   ucs_status_t status;
   uct_iface_config_t *config;
   uct_iface_h iface;
 
   UCXCHECK(uct_md_iface_config_read(md, tl->tl_name, NULL, NULL, &config),
            return NULL, "read MD iface config for TL '%s'", tl->tl_name);
 
   status = uct_config_modify(config, "IB_TX_INLINE_RESP", "0");
   if (status != UCS_OK) {
       WARN("Failed to modify MD configuration for '%s', error %s",
            tl->tl_name, ucs_status_string(status));
       uct_config_release(config);
       return NULL;
   }
 
   params.field_mask =
       UCT_IFACE_PARAM_FIELD_OPEN_MODE | UCT_IFACE_PARAM_FIELD_DEVICE |
       UCT_IFACE_PARAM_FIELD_STATS_ROOT | UCT_IFACE_PARAM_FIELD_RX_HEADROOM;
   params.open_mode            = UCT_IFACE_OPEN_MODE_DEVICE;
   params.mode.device.tl_name  = tl->tl_name;
   params.mode.device.dev_name = tl->dev_name;
   params.stats_root           = NULL;
   params.rx_headroom          = 0;
 
   status = uct_iface_open(md, worker, &params, config, &iface);
   uct_config_release(config);
   UCXCHECK(status, return NULL, "open UCT iface %s/%s",
                       tl->tl_name, tl->dev_name);
 
   uct_iface_progress_enable(iface, UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
 
   return iface;
 }
static uct_iface_h
flagcx_uct_md_iface_open(uct_worker_h worker, uct_component_h comp,
                       unsigned md_index, const char *md_name,
                       const char *tl_name, const char *dev_name,
                       uct_md_h *md_p) {
  uct_iface_h iface = NULL;
  ucs_status_t status;
  uct_md_config_t *md_config;
  uct_md_h md;
  uct_md_attr_t md_attr;
  uct_tl_resource_desc_t *tls;
  unsigned tls_count, i;

  UCXCHECK(uct_md_config_read(comp, NULL, NULL, &md_config), return NULL,
           "read MD[%d] config", md_index);

  status = uct_md_open(comp, md_name, md_config, &md);
  uct_config_release(md_config);
  UCXCHECK(status, return NULL, "open MD[%d/%s]", md_index, md_name);

  UCXCHECK(uct_md_query(md, &md_attr), goto out, "query MD[%d/%s]", md_index,
           md_name);

  UCXCHECK(uct_md_query_tl_resources(md, &tls, &tls_count), goto out,
           "query resources MD[%d/%s]", md_index, md_name);

  for (i = 0; i < tls_count; i++) {
    if (!strcmp(dev_name, tls[i].dev_name) &&
        !strcmp(tl_name, tls[i].tl_name)) {

      iface = flagcx_uct_resource_iface_open(worker, md, &tls[i]);
      break;
    }
  }

  uct_release_tl_resource_list(tls);

 out:
  if (iface == NULL) {
    uct_md_close(md);
  } else {
    *md_p = md;
  }
  return iface;
}

static flagcx_uct_iface_t *flagcx_uct_iface_open(flagcx_uct_worker_t *uct_worker,
                                              const char *tl_name,
                                              const char *dev_name) {
  uct_worker_h worker         = uct_worker->worker;
  flagcx_uct_iface_t *uct_iface = NULL;
  uct_iface_h iface           = NULL;
  uct_component_h *comps, *comp;
  unsigned comps_count, i;
  uct_component_attr_t comp_attr;
  uct_iface_attr_t iface_attr;
  uct_md_h md;
  uct_md_attr_t md_attr;

  UCXCHECK(uct_query_components(&comps, &comps_count), return NULL,
           "query component list");

  for (comp = comps; comp < comps + comps_count; comp++) {
    comp_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT;
    UCXCHECK(uct_component_query(*comp, &comp_attr), goto out,
             "query component");

    comp_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
    comp_attr.md_resources =
        (uct_md_resource_desc_t*)alloca(sizeof(*comp_attr.md_resources) * comp_attr.md_resource_count);
    UCXCHECK(uct_component_query(*comp, &comp_attr), goto out,
             "query component resources");

    for (i = 0; i < comp_attr.md_resource_count; i++) {
      iface = flagcx_uct_md_iface_open(worker, *comp, i,
                                     comp_attr.md_resources[i].md_name, tl_name,
                                     dev_name, &md);
      if (iface != NULL) {
        goto found;
      }
    }
  }

  if (iface == NULL) {
    WARN("Failed to open iface for tl_name=%s dev_name=%s", tl_name, dev_name);
    goto out;
  }

found:
  UCXCHECK(uct_iface_query(iface, &iface_attr), goto fail,
           "iface for tl_name=%s dev_name=%s", tl_name, dev_name);

  if (!(iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP)) {
    WARN("Interface flag CONNECT_TO_EP is not set");
    goto fail;
  }

  if (!(iface_attr.cap.flags & UCT_IFACE_FLAG_GET_ZCOPY) ||
      !(iface_attr.cap.flags & UCT_IFACE_FLAG_PUT_ZCOPY)) {
    WARN("Interface does not support ZCOPY (flags=0x%lx)", iface_attr.cap.flags);
    goto fail;
  }

  UCXCHECK(uct_md_query(md, &md_attr), goto fail, "query md for iface %p",
           iface);

  uct_iface = (flagcx_uct_iface_t*)calloc(1, sizeof(*uct_iface));
  if (uct_iface == NULL) {
    WARN("Failed to alloc uct iface structure");
    goto fail;
  }

  uct_iface->ep_addr_size     = iface_attr.ep_addr_len;
  uct_iface->md               = md;
  uct_iface->comp             = *comp;
  uct_iface->rkey_packed_size = md_attr.rkey_packed_size;

  if (iface_attr.cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
    uct_iface->am_max_short = iface_attr.cap.am.max_short;
  }

  if (iface_attr.cap.flags & UCT_IFACE_FLAG_GET_ZCOPY) {
    uct_iface->min_get_zcopy = iface_attr.cap.get.min_zcopy;
  }

  // Note: context variable not available here, using placeholder values
  if (64 > uct_iface->am_max_short) { // context.am_short_size placeholder
    WARN("Failed RTR does not fit in face AM short (%uB > %zuB)",
         64, uct_iface->am_max_short);
    goto fail;
  }

  if (uct_iface->rkey_packed_size > 64) { // context.rkey_size placeholder
    WARN("Interface rkey_packed_size %zu too big", uct_iface->rkey_packed_size);
    goto fail;
  }

  if (iface_attr.device_addr_len > 0) {
    uct_iface->dev_addr_size = iface_attr.device_addr_len;
    uct_iface->dev_addr      = (uct_device_addr_t*)calloc(1, iface_attr.device_addr_len);
    if (uct_iface->dev_addr == NULL) {
      WARN("Failed to alloc dev_addr");
      goto fail;
    }

    UCXCHECK(uct_iface_get_device_address(iface, (uct_device_addr_t*)uct_iface->dev_addr),
             goto fail, "query iface device addr for tl_name=%s dev_name=%s",
             tl_name, dev_name);
  }

  if (iface_attr.iface_addr_len > 0) {
    uct_iface->addr_size = iface_attr.iface_addr_len;
    uct_iface->addr      = (uct_iface_addr_t*)calloc(1, iface_attr.iface_addr_len);
    if (uct_iface->addr == NULL) {
      WARN("Failed to alloc iface addr");
      goto fail;
    }

    UCXCHECK(uct_iface_get_address(iface, (uct_iface_addr_t*)uct_iface->addr), goto fail,
             "query iface addr to tl_name=%s dev_name=%s", tl_name, dev_name);
  }

  uct_iface->iface = iface;

out:
  uct_release_component_list(comps);
  return uct_iface;

fail:
  if (uct_iface != NULL) {
    free(uct_iface->dev_addr);
    free(uct_iface->addr);
    free(uct_iface);
  }
  if (iface != NULL) {
    uct_iface_close(iface);
  }
  uct_release_component_list(comps);
  return NULL;
}



  
static void flagcx_uct_iface_close(flagcx_uct_iface_t *uct_iface) {
  uct_iface_close(uct_iface->iface);
  uct_md_close(uct_iface->md);
  free(uct_iface->dev_addr);
  free(uct_iface->addr);
  free(uct_iface);
}

static flagcx_uct_worker_t *flagcx_uct_worker_create(flagcx_uct_context_t *context,
                                                 int dev) {
  flagcx_uct_worker_t *w = NULL;
  flagcxResult_t result;

      w = (flagcx_uct_worker_t*)calloc(1, sizeof(*w));
  if (w == NULL) {
    WARN("Failed worker allocation: dev=%d", dev);
    return NULL;
  }

  UCXCHECK(ucs_async_context_create(UCS_ASYNC_MODE_THREAD_SPINLOCK, &w->async),
           goto fail_free, "create UCT async context: dev=%d", dev);
  UCXCHECK(uct_worker_create(w->async, UCS_THREAD_MODE_SINGLE, &w->worker),
           goto fail_context, "create UCT worker: dev=%d", dev);

  w->id.dev    = dev;
  w->id.thread = pthread_self();
  w->context   = context;

  w->uct_iface = flagcx_uct_iface_open(w, context->tl_name, flagcx_dev_name(dev));
  if (w->uct_iface == NULL) {
    WARN("Failed to create UCT iface for worker: dev=%d", dev);
    goto fail;
  }

  result = context->ops.iface_set(w->uct_iface);
  if (result != flagcxSuccess) {
    WARN("Failed to set iface configuration: dev=%d", dev);
    goto fail;
  }

  w->next              = context->worker_list;
  context->worker_list = w;
  return w;

fail:
  if (w->uct_iface != NULL) {
    flagcx_uct_iface_close(w->uct_iface);
  }

  uct_worker_destroy(w->worker);
fail_context:
  ucs_async_context_destroy(w->async);
fail_free:
  free(w);
  return NULL;
}
static flagcx_uct_worker_t *flagcx_uct_worker_get(flagcx_uct_context_t *context,
                                             int dev) {
 flagcx_uct_worker_t *w;
 pthread_mutex_lock(&flagcx_uct_lock);
 for (w = context->worker_list; w != NULL; w = w->next) {
   if (w->id.dev == dev) {
     goto found;
   }
 }
 w = flagcx_uct_worker_create(context, dev);
 if (w == NULL) {
   goto out;
 }
found:
  w->count++;
out:
  pthread_mutex_unlock(&flagcx_uct_lock);
  return w;
}

// Additional macros
// FLAGCXCHECKGOTO already defined in check.h


#define PTHREADCHECKGOTO(statement, message, ret, label) \
  do { \
    int _ret = statement; \
    if (_ret != 0) { \
      WARN("Failed: " message ": %s", strerror(_ret)); \
      ret = flagcxSystemError; \
      goto label; \
    } \
  } while (0)

// External variable declarations
pthread_mutex_t flagcx_p2p_lock = PTHREAD_MUTEX_INITIALIZER;
#ifdef HAVE_SHARP_PLUGIN
extern int flagcxNSharpDevs;
#else
/* In case sharp plugin is not there just define this variable locally to make code cleaner */
int flagcxNSharpDevs;
#endif
flagcxDebugLogger_t pluginLogFunction;

// Forward declaration for IB device structure
typedef struct flagcxIbDev flagcxIbDev;

// External function declarations
extern flagcxResult_t flagcx_p2p_ib_init(int *nDevs, int *nmDevs, void *flagcxIbDevs, char *flagcxIbIfName, union flagcxSocketAddress *flagcxIbIfAddr, pthread_t *flagcxIbAsyncThread, void *logFunction);

// Additional structures will be defined after the main structures

// Helper macros
#define flagcx_uct_wr_comm_get(base) ((flagcx_uct_wr_comm_t*)(base))
// ucs_list macros already defined in UCX headers

// Additional device structure


// UCX UCT memory handle
typedef struct {
  uct_mem_h         memh;
  struct flagcx_uct_comm *comm;
  uct_rkey_bundle_t bundle;
  uint8_t           rkey[];
} flagcx_uct_memh_t;

// Additional structures
typedef struct {
    int state;
    flagcx_uct_comm_t *comm;
    int offset;
    int ready;
} flagcx_uct_stage_t;

typedef struct {
    uint64_t magic;
    struct {
        int id;
        union flagcxSocketAddress addr;
    } listener;
    flagcx_uct_stage_t stage;
    flagcx_uct_comm_t *comm;
} flagcx_uct_listen_handle_t;

typedef struct {
    struct flagcxSocket sock;
    struct flagcx_uct_context *context;
    int dev;
    int id;
    struct flagcx_uct_worker *uct_worker;
    struct flagcx_uct_comm *comm;
    flagcx_uct_stage_t stage;
} flagcx_uct_listen_comm_t;


// Global context
static flagcx_uct_context_t context = {
    .tl_name   = "rc_mlx5",
    .dev_count = -1,
    .merge_dev_count = -1
};

// Global variables

typedef enum {
  FLAGCX_UCT_REQ_IRECV  = -1,
  FLAGCX_UCT_REQ_IFLUSH = -2
} flagcx_uct_request_type_t;

struct flagcx_uct_rdesc;

/* On-the-wire descriptor of a posted receive request entry */
typedef struct {
  int        tag;
  int        size;
  void       *data;
  int        matched;
  uct_rkey_t rkey;
} flagcx_uct_chunk_t;

/* On-the-wire descriptor of a receive request containing many chunks */
typedef struct {
  uint64_t              id;
  uint16_t              count;
  uint32_t              size;
  struct flagcx_uct_rdesc *peer_rdesc; /* Acts as a cookie along with id */
  flagcx_uct_chunk_t      chunk[FLAGCX_UCX_UCT_MAX_RECVS];
} flagcx_uct_rdesc_hdr_t;

/* On-the-wire descriptor for receive request completion */
typedef struct {
  uint64_t              id;
  struct flagcx_uct_rdesc *rdesc;
  int                   count; /* Number of sizes contained */
  int                   sizes[FLAGCX_UCX_UCT_MAX_RECVS];
} flagcx_uct_atp_t;

/*
 * flagcx local request handler to progress:
 * - size -1 for multi receive
 * - size -2 for flush
 * - size > 0 for send
 */
typedef struct {
  /* Pending GET (iflush) PUT (isend) or receiving one ATP (irecv) */
  uct_completion_t      completion;
  int                   size;
  struct flagcx_uct_rdesc *rdesc;
} flagcx_uct_req_t;

/* Pending receive descriptor either on the receive or sending side */
typedef struct flagcx_uct_rdesc {
  int                   flagcx_usage; /* flagcx requests not finished/started */
  int                   send_atp;   /* >1 pending isend, ==1 pending atp send */

  union {
    ucs_list_link_t       list;  /* comm's linked list */
    struct flagcx_uct_rdesc *next; /* inserted in free list */
  };

  struct flagcx_uct_wr_comm  *comm;
  flagcx_uct_rdesc_hdr_t     desc;
  flagcx_uct_req_t           reqs[FLAGCX_UCX_UCT_MAX_RECVS];    /* flagcx requests */
  int                      sizes[FLAGCX_UCX_UCT_MAX_RECVS];   /* ATP received sizes */
  flagcx_uct_chunk_t         storage[FLAGCX_UCX_UCT_MAX_RECVS]; /* Don't use directly */
} flagcx_uct_rdesc_t;

typedef struct flagcx_uct_wr_comm {
  flagcx_uct_comm_t      base;

  int                  rdesc_alloc; /* Track allocated rdescs */
  flagcx_uct_rdesc_t     *free_rdesc; /* Available rdesc for reuse */
  uint64_t             rdesc_id;    /* Next sequence number to use */

  /* Received RTRs: used by Sender communicator in ->isend() */
  ucs_list_link_t      rdesc_list;

} flagcx_uct_wr_comm_t;

// ============================================================================
// UCX UCT Core Implementation Functions (from net_ucx_uct.cc)
// ============================================================================

void flagcx_uct_empty_callback(uct_completion_t *comp) {
  assert(comp->count == 0);
}

static flagcx_uct_rdesc_t *flagcx_uct_comm_rdesc_get(flagcx_uct_wr_comm_t *comm) {
  flagcx_uct_rdesc_t *rdesc = comm->free_rdesc;

  if (rdesc == NULL) {
    rdesc = (flagcx_uct_rdesc_t*)calloc(1, sizeof(*rdesc));
  } else {
    comm->free_rdesc = rdesc->next;
  }

  rdesc->next = NULL;
  rdesc->comm = comm;
  comm->rdesc_alloc++;
  return rdesc;
}

static size_t flagcx_uct_rdesc_size(int n) {
  return n * sizeof(flagcx_uct_chunk_t) + sizeof(flagcx_uct_rdesc_hdr_t);
}

/* Prepare a receive descriptor from irecv()/iflush() side */
static void flagcx_uct_rdesc_set(flagcx_uct_rdesc_t *rdesc, uint64_t id, int n,
                               void **data, size_t *sizes, int *tags,
                               flagcx_uct_memh_t **uct_memh) {
  flagcx_uct_rdesc_hdr_t *desc = &rdesc->desc;
  int i;

  /* Populate header */
  desc->id         = id;
  desc->count      = n;
  desc->size       = flagcx_uct_rdesc_size(n);
  desc->peer_rdesc = rdesc; /* cookie, will be returned in ATP */

  /* Ref count that prevents flagcx from releasing memory */
  rdesc->flagcx_usage = 1;
  rdesc->send_atp   = 0;

  /* Zero (iflush) or one or many receive request are contained */
  for (i = 0; i < n; i++) {
    desc->chunk[i].tag     = tags[i];
    desc->chunk[i].size    = sizes[i];
    desc->chunk[i].data    = data[i];
    desc->chunk[i].matched = 0;
    desc->chunk[i].rkey    = uct_memh[i]->bundle.rkey;
  }
}

static flagcx_uct_req_t *flagcx_uct_rdesc_get_req(flagcx_uct_rdesc_t *rdesc, int i,
                                              int size) {
  flagcx_uct_req_t *req;

  assert(i < FLAGCX_UCX_UCT_MAX_RECVS);

  req        = &rdesc->reqs[i];
  req->size  = size;
  req->rdesc = rdesc;

  req->completion.func   = flagcx_uct_empty_callback;
  req->completion.count  = 1;
  req->completion.status = UCS_OK;

  return &rdesc->reqs[i];
}

static void flagcx_uct_comm_rdesc_put(flagcx_uct_rdesc_t *rdesc) {
  flagcx_uct_wr_comm_t *comm = rdesc->comm;

  assert(comm != NULL);

  rdesc->desc.id   = -1;
  rdesc->comm      = NULL;
  rdesc->next      = comm->free_rdesc;
  comm->free_rdesc = rdesc;
  comm->rdesc_alloc--;
}

/* On receiver side, after ->irecv(), expect corresponding ATP */
static ucs_status_t flagcx_uct_atp_callback(void *arg, void *data, size_t length,
                                          unsigned flags) {
  flagcx_uct_atp_t *atp = (flagcx_uct_atp_t*)((uint8_t*)data + 8);

  assert(length == (sizeof(*atp) + 8));
  assert(*(flagcx_uct_comm_t**)data == &atp->rdesc->comm->base);
  assert(atp->id == atp->rdesc->desc.id);
  assert(atp->count == atp->rdesc->desc.count);
  assert(atp->rdesc->reqs[0].completion.count == 1);

  atp->rdesc->reqs[0].completion.count--;
  memcpy(atp->rdesc->sizes, atp->sizes, atp->count * sizeof(*atp->sizes));
  return UCS_OK;
}

/* On sender side, asynchronously receive rdesc/RTR, later used by ->isend() */
static ucs_status_t flagcx_uct_rtr_callback(void *arg, void *data, size_t length,
                                          unsigned flags) {
  flagcx_uct_comm_t *base_comm = *(flagcx_uct_comm_t **)data;
  flagcx_uct_wr_comm_t *comm   = flagcx_uct_wr_comm_get(base_comm);
  flagcx_uct_rdesc_hdr_t *desc = (flagcx_uct_rdesc_hdr_t*)((uint8_t*)data + 8);
  size_t size                = desc->size;
  flagcx_uct_rdesc_t *rdesc;

  rdesc = flagcx_uct_comm_rdesc_get(comm);
  if (rdesc == NULL) {
    WARN("Failed to get an rdesc in RTR callback");
    return UCS_ERR_NO_MEMORY; /* Cannot happend */
  }

  ucs_list_add_tail(&comm->rdesc_list, &rdesc->list);

  assert((size + 8) == length);
  assert(size == flagcx_uct_rdesc_size(desc->count));

  memcpy(&rdesc->desc, desc, size);
  rdesc->flagcx_usage = desc->count;
  rdesc->send_atp   = desc->count + 1;
  return UCS_OK;
}

static void flagcx_uct_send_atp(flagcx_uct_wr_comm_t *comm,
                              flagcx_uct_rdesc_t *rdesc) {
  ucs_status_t status;
  flagcx_uct_atp_t atp;
  int i;

  assert(rdesc->send_atp == 1);

    status = uct_ep_fence(comm->base.uct_ep->ep, 0);
    if (status != UCS_OK) {
      return;
    }

    atp.id    = rdesc->desc.id;
    atp.rdesc = rdesc->desc.peer_rdesc;
    atp.count = rdesc->desc.count;

    /* Sizes from isend() are lower or equal to their irecv() side */
    for (i = 0; i < rdesc->desc.count; i++) {
      atp.sizes[i] = rdesc->reqs[i].size;
    }

    status = uct_ep_am_short(comm->base.uct_ep->ep, FLAGCX_UCT_AM_ATP,
                             (uint64_t)comm->base.remote.comm, &atp, sizeof(atp));
    if (status == UCS_OK) {
      rdesc->send_atp = 0;
  }
}

flagcxResult_t flagcx_p2p_gdr_support()
{
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, ibGdrSupportInitOnce);
  if (!flagcxIbGdrModuleLoaded)
    return flagcxSystemError;
  return flagcxSuccess;
}
static flagcxResult_t flagcx_uct_send(flagcx_uct_wr_comm_t *comm, void *data,
                                  int size, flagcx_uct_memh_t *uct_memh,
                                  flagcx_uct_rdesc_t *rdesc, int i,
                                  void **request) {
  ucs_status_t status;
  uct_iov_t iov;
  flagcx_uct_req_t *req;

  *request = NULL;

  if (comm->base.uct_ep && comm->base.uct_ep->ep) {
    /* Details for local data */
    iov.buffer = data;
    iov.length = size;
    iov.memh   = uct_memh->memh;
    iov.stride = iov.length;
    iov.count  = 1;

    assert(size <= rdesc->desc.chunk[i].size);

    req = flagcx_uct_rdesc_get_req(rdesc, i, size); /* flagcx request */

    status = uct_ep_put_zcopy(comm->base.uct_ep->ep, &iov, 1,
                              (uint64_t)rdesc->desc.chunk[i].data,
                              rdesc->desc.chunk[i].rkey, &req->completion);

    if (status == UCS_OK) {
      req->completion.count--;
    } else if (status != UCS_INPROGRESS) {
      return flagcxSuccess;
    }

    rdesc->desc.chunk[i].matched = 1;
    --rdesc->send_atp;

    if (rdesc->send_atp == 1) {
      ucs_list_del(&rdesc->list); /* all ->isend() were now matched */
      flagcx_uct_send_atp(comm, rdesc);
    }

    *request = req;
  }
  return flagcxSuccess;
}

// ============================================================================
// UCX UCT Library Functions (from ucx_uct_lib.cc)
// ============================================================================

flagcxResult_t flagcx_uct_iface_set_handler(flagcx_uct_iface_t *uct_iface,
                                         int id, uct_am_callback_t callback) {
  UCXCHECK(uct_iface_set_am_handler(uct_iface->iface, id, callback, NULL, 0),
           return flagcxInternalError, "get AM handler id=%d", id);
  return flagcxSuccess;
}



void flagcx_uct_comm_deinit(flagcx_uct_comm_t *comm) {
  if (comm->uct_ep != NULL) {
    uct_ep_destroy(comm->uct_ep->ep);
    free(comm->uct_ep);
  }
  if (comm->uct_worker != NULL) {
    free(comm->uct_worker);
  }
}

int flagcx_uct_flush_index(flagcx_uct_comm_t *base, int *sizes, int n) {
  int last = -1;
  int i;

  for (i = 0; i < n; i++) {
    if (sizes[i]) {
      last = i;
    }
  }

  return last;
}

flagcxResult_t flagcx_uct_flush(flagcx_uct_comm_t *base_comm, void *data, int size,
                             flagcx_uct_memh_t *uct_memh,
                             uct_completion_t *completion, void **request) {
  // Simple flush implementation
  *request = NULL;
  return flagcxSuccess;
}

// ============================================================================
// UCX UCT Plugin Functions (from net_ucx_uct.cc)
// ============================================================================

static flagcxResult_t flagcx_uct_wr_isend(void *send_comm, void *data, size_t size,
                                      int tag, void *mhandle,  void* phandle, void **request) {
  flagcx_uct_wr_comm_t *comm = flagcx_uct_wr_comm_get((flagcx_uct_comm_t*)send_comm);
  flagcx_uct_rdesc_t *rdesc;
  int i;

  *request = NULL;

  ucs_list_for_each(rdesc, &comm->rdesc_list, list) {
    for (i = 0; i < rdesc->desc.count; i++) {
      if (rdesc->desc.chunk[i].matched || (rdesc->desc.chunk[i].tag != tag)) {
        continue;
      }

      return flagcx_uct_send(comm, data, size, (flagcx_uct_memh_t*)mhandle, rdesc, i, request);
    }
  }

  /* Progress here to make sure we receive non-solicited RTRs */
  if (comm->base.uct_worker && comm->base.uct_worker->worker) {
    uct_worker_progress(comm->base.uct_worker->worker);
  }
  return flagcxSuccess;
}


static flagcxResult_t flagcx_uct_wr_test(void *request, int *done, int *sizes) {
  flagcx_uct_req_t *req      = (flagcx_uct_req_t*)request;
  flagcx_uct_rdesc_t *rdesc  = req->rdesc;
  flagcx_uct_wr_comm_t *comm = rdesc->comm;

  if (comm->base.uct_worker && comm->base.uct_worker->worker) {
    uct_worker_progress(comm->base.uct_worker->worker);
  }

  *done = 0;

  if (rdesc->send_atp == 1) {
    /* Slowpath */
    flagcx_uct_send_atp(comm, rdesc);

    if (rdesc->send_atp && rdesc->flagcx_usage == 1) {
      /* Keep the last isend request until ATP is out */
      return flagcxSuccess;
    }
  }

  if (req->completion.count > 0) {
    return flagcxSuccess;
  }

  *done = 1;

  if (req->size == FLAGCX_UCT_REQ_IRECV) {
    assert(&rdesc->reqs[0] == req);
    if (sizes != NULL) {
      memcpy(sizes, rdesc->sizes, rdesc->desc.count * sizeof(*sizes));
    }
  } else if (req->size == FLAGCX_UCT_REQ_IFLUSH) {
    assert(&rdesc->reqs[0] == req);
  } else {
    /* ->isend() request */
    assert(req->size > -1);
    if (sizes != NULL) {
      sizes[0] = req->size;
    }
  }

  if (--rdesc->flagcx_usage < 1) {
    assert(rdesc->send_atp == 0);
    assert(rdesc->flagcx_usage == 0);
    flagcx_uct_comm_rdesc_put(rdesc);
  }

  return flagcxSuccess;
}

flagcxResult_t flagcx_p2p_ib_init(int *nDevs, int *nmDevs, flagcxIbDev *flagcxIbDevs, char *flagcxIbIfName, union flagcxSocketAddress *flagcxIbIfAddr, pthread_t *flagcxIbAsyncThread, flagcxDebugLogger_t logFunction)
{
  flagcxResult_t ret = flagcxSuccess;
  int flagcxNIbDevs = *nDevs;
  int flagcxNMergedIbDevs = *nmDevs;
  pluginLogFunction = logFunction;
  if (flagcxNIbDevs == -1) {
    for (int i=0; i< MAX_IB_DEVS; i++)
      onces[i].once = PTHREAD_ONCE_INIT;
    pthread_mutex_lock(&flagcx_p2p_lock);
    flagcxWrapIbvForkInit();
    if (flagcxNIbDevs == -1) {
      int nIpIfs = 0;
      flagcxNIbDevs = 0;
      flagcxNMergedIbDevs = 0;
      flagcxNSharpDevs = 0;
      nIpIfs = flagcxFindInterfaces(flagcxIbIfName, flagcxIbIfAddr, MAX_IF_NAME_SIZE, 1);
      if (nIpIfs != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = flagcxInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;
      // Check if user defined which IB device:port to use
      const char* userIbEnv = flagcxGetEnv("FLAGCX_IB_HCA");
      struct netIf userIfs[MAX_IB_DEVS];
      int searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      int searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (flagcxSuccess != flagcxWrapIbvGetDeviceList(&devices, &nIbDevs)) { ret = flagcxInternalError; goto fail; }
      for (int d=0; d<nIbDevs && flagcxNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context;
        if (flagcxSuccess != flagcxWrapIbvOpenDevice(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        enum flagcxIbProvider ibProvider = IB_PROVIDER_NONE;
        char dataDirectDevicePath[PATH_MAX];
        int dataDirectSupported = 0;
        int skipNetDevForDataDirect = 0;
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        if (flagcxSuccess != flagcxWrapIbvQueryDevice(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (flagcxSuccess != flagcxWrapIbvCloseDevice(context)) { ret = flagcxInternalError; goto fail; }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
          for (int dataDirect = skipNetDevForDataDirect; dataDirect < 1 + dataDirectSupported; ++dataDirect) {
            struct ibv_port_attr portAttr;
            uint32_t portSpeed;
            if (flagcxSuccess != flagcxWrapIbvQueryPort(context, port_num, &portAttr)) {
              WARN("NET/IB : Unable to query port_num %d", port_num);
              continue;
            }
            if (portAttr.state != IBV_PORT_ACTIVE) continue;
            if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
                && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

            // check against user specified HCAs/ports
            if (! (matchIfList(devices[d]->name, port_num, userIfs, nUserIfs, searchExact) ^ searchNot)) {
              continue;
            }
            pthread_mutex_init(&flagcxIbDevs[flagcxNIbDevs].lock, NULL);
            flagcxIbDevs[flagcxNIbDevs].device = d;
            flagcxIbDevs[flagcxNIbDevs].ibProvider = ibProvider;
            flagcxIbDevs[flagcxNIbDevs].guid = devAttr.sys_image_guid;
            flagcxIbDevs[flagcxNIbDevs].portAttr = portAttr;
            flagcxIbDevs[flagcxNIbDevs].portNum = port_num;
            flagcxIbDevs[flagcxNIbDevs].link = portAttr.link_layer;
            #if HAVE_STRUCT_IBV_PORT_ATTR_ACTIVE_SPEED_EX
            portSpeed = portAttr.active_speed_ex ? portAttr.active_speed_ex : portAttr.active_speed;
            #else
            portSpeed = portAttr.active_speed;
            #endif
            flagcxIbDevs[flagcxNIbDevs].speed = flagcx_p2p_ib_speed(portSpeed) * flagcx_p2p_ib_width(portAttr.active_width);
            flagcxIbDevs[flagcxNIbDevs].context = context;
            flagcxIbDevs[flagcxNIbDevs].pdRefs = 0;
            flagcxIbDevs[flagcxNIbDevs].pd = NULL;
            if (!dataDirect) {
              strncpy(flagcxIbDevs[flagcxNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
              FLAGCXCHECKGOTO(flagcx_p2p_ib_pci_path(flagcxIbDevs, flagcxNIbDevs, flagcxIbDevs[flagcxNIbDevs].devName, &flagcxIbDevs[flagcxNIbDevs].pciPath, &flagcxIbDevs[flagcxNIbDevs].realPort), ret, fail);
            } else {
              snprintf(flagcxIbDevs[flagcxNIbDevs].devName, MAXNAMESIZE, "%s_dma", devices[d]->name);
              flagcxIbDevs[flagcxNIbDevs].pciPath = (char*)malloc(PATH_MAX);
              strncpy(flagcxIbDevs[flagcxNIbDevs].pciPath, dataDirectDevicePath, PATH_MAX);
              flagcxIbDevs[flagcxNIbDevs].capsProvider.mlx5.dataDirect = 1;
            }
            flagcxIbDevs[flagcxNIbDevs].maxQp = devAttr.max_qp;
            flagcxIbDevs[flagcxNIbDevs].mrCache.capacity = 0;
            flagcxIbDevs[flagcxNIbDevs].mrCache.population = 0;
            flagcxIbDevs[flagcxNIbDevs].mrCache.slots = NULL;
            FLAGCXCHECK(flagcxIbStatsInit(&flagcxIbDevs[flagcxNIbDevs].stats));

          // Enable ADAPTIVE_ROUTING by default on IB networks
            // But allow it to be overloaded by an env parameter
            flagcxIbDevs[flagcxNIbDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
            if (flagcxParamIbAdaptiveRouting() != -2) flagcxIbDevs[flagcxNIbDevs].ar = flagcxParamIbAdaptiveRouting();

            flagcxIbDevs[flagcxNIbDevs].isSharpDev = 0;
            if (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND)
            {
              flagcxIbDevs[flagcxNIbDevs].isSharpDev = 1;
              flagcxIbDevs[flagcxNIbDevs].maxQp = flagcxParamSharpMaxComms();
              flagcxNSharpDevs++;
            }
            TRACE(FLAGCX_NET,"NET/IB: [%d] %s:%s:%d/%s provider=%s speed=%d context=%p pciPath=%s ar=%d", d, devices[d]->name, devices[d]->dev_name, flagcxIbDevs[flagcxNIbDevs].portNum,
              FLAGCX_IB_LLSTR(portAttr.link_layer), ibProviderName[flagcxIbDevs[flagcxNIbDevs].ibProvider], flagcxIbDevs[flagcxNIbDevs].speed, context, flagcxIbDevs[flagcxNIbDevs].pciPath, flagcxIbDevs[flagcxNIbDevs].ar);
            if (flagcxIbAsyncThread != NULL) {
              PTHREADCHECKGOTO(pthread_create(flagcxIbAsyncThread, NULL, flagcxIbAsyncThreadMain, flagcxIbDevs + flagcxNIbDevs), "pthread_create", ret, fail);
              flagcxSetThreadName(*flagcxIbAsyncThread, "flagcx IbAsync %2d", flagcxNIbDevs);
              PTHREADCHECKGOTO(pthread_detach(*flagcxIbAsyncThread), "pthread_detach", ret, fail); // will not be pthread_join()'d
            }

            // Add this plain physical device to the list of virtual devices
            int vDev;
            flagcxNetVDeviceProps_t vProps = {0};
            vProps.ndevs = 1;
            vProps.devs[0] = flagcxNIbDevs;
            FLAGCXCHECK(flagcxIbMakeVDeviceInternal(&vDev, &vProps, flagcxNIbDevs, &flagcxNMergedIbDevs));

            flagcxNIbDevs++;
            nPorts++;
          }
        }
        if (nPorts == 0 && flagcxSuccess != flagcxWrapIbvCloseDevice(context))  { ret = flagcxInternalError; goto fail; }
      }
      
      if (nIbDevs && (flagcxSuccess != flagcxWrapIbvFreeDeviceList(devices))) { ret = flagcxInternalError; goto fail; };
    }
    if (flagcxNIbDevs == 0) {
      INFO(FLAGCX_INIT|FLAGCX_NET, "NET/IB : No device found.");
    }

    // Print out all net devices to the user (in the same format as before)
    char line[2048];
    line[0] = '\0';
    // Determine whether RELAXED_ORDERING is enabled and possible
#ifdef HAVE_IB_PLUGIN
    flagcxIbRelaxedOrderingEnabled = flagcxIbRelaxedOrderingCapable();
#endif
    for (int d = 0; d < flagcxNIbDevs; d++) {
#ifdef HAVE_SHARP_PLUGIN
            snprintf(line+strlen(line), sizeof(line)-strlen(line), " [%d]%s:%d/%s%s", d, flagcxIbDevs[d].devName,
              flagcxIbDevs[d].portNum, FLAGCX_IB_LLSTR(flagcxIbDevs[d].link),
              flagcxIbDevs[d].isSharpDev ? "/SHARP" : "");
#else
      snprintf(line+strlen(line), sizeof(line)-strlen(line), " [%d]%s:%d/%s", d, flagcxIbDevs[d].devName,
        flagcxIbDevs[d].portNum, FLAGCX_IB_LLSTR(flagcxIbDevs[d].link));
#endif
    }
    char addrline[SOCKET_NAME_MAXLEN+1];
    INFO(FLAGCX_INIT|FLAGCX_NET, "NET/IB : Using%s %s; OOB %s:%s", line, 
#ifdef HAVE_IB_PLUGIN
      flagcxIbRelaxedOrderingEnabled ? "[RO]" : ""
#else
      ""
#endif
      ,
      flagcxIbIfName, flagcxSocketToString(flagcxIbIfAddr, addrline, 1));
    *nDevs = flagcxNIbDevs;
    *nmDevs = flagcxNMergedIbDevs;
    pthread_mutex_unlock(&flagcx_p2p_lock);
  }
exit:
  return ret;
fail:
  pthread_mutex_unlock(&flagcx_p2p_lock);
  goto exit;
}


flagcxResult_t flagcx_uct_comm_init(flagcx_uct_comm_t *comm,
                                 flagcx_uct_context_t *context,
                                 flagcx_uct_worker_t *worker, int dev,
                                 const flagcx_uct_comm_t *remote_comm) {
  if (worker == NULL) {
    worker = flagcx_uct_worker_get(context, dev);
  }

  comm->uct_worker = worker;
  if (comm->uct_worker == NULL) {
    return flagcxSystemError;
  }

  comm->dev         = dev;
  comm->context     = context;
  comm->remote.comm = remote_comm;
  comm->uct_iface   = comm->uct_worker->uct_iface;
  comm->uct_ep      = flagcx_uct_ep_create(comm->uct_iface);
  if (comm->uct_ep == NULL) {
    return flagcxSystemError;
  }

  return flagcxSuccess;
}
static flagcxResult_t flagcx_uct_wr_comm_init(flagcx_uct_comm_t *base_comm,
                                          flagcx_uct_context_t *context,
                                          flagcx_uct_worker_t *worker, int dev,
                                          const flagcx_uct_comm_t *remote_comm) {
  flagcx_uct_wr_comm_t *comm = flagcx_uct_wr_comm_get(base_comm);

  ucs_list_head_init(&comm->rdesc_list);
  return flagcx_uct_comm_init(&comm->base, context, worker, dev, remote_comm);
}

static flagcxResult_t flagcx_uct_wr_comm_alloc(flagcx_uct_comm_t **comm_p) {
  flagcx_uct_wr_comm_t *comm = (flagcx_uct_wr_comm_t*)calloc(1, sizeof(flagcx_uct_wr_comm_t));
  if (comm != NULL) {
    *comm_p = &comm->base;
    return flagcxSuccess;
  }

  return flagcxSystemError;
}

static flagcxResult_t flagcx_uct_wr_iface_set(flagcx_uct_iface_t *uct_iface) {
  FLAGCXCHECK(flagcx_uct_iface_set_handler(uct_iface, FLAGCX_UCT_AM_RTR,
                                       flagcx_uct_rtr_callback));
  FLAGCXCHECK(flagcx_uct_iface_set_handler(uct_iface, FLAGCX_UCT_AM_ATP,
                                       flagcx_uct_atp_callback));
  return flagcxSuccess;
}

static const uct_device_addr_t *
flagcx_uct_ep_addr_dev(const flagcx_uct_ep_addr_t *addr) {
  return (uct_device_addr_t*)addr->data;
}

static const uct_ep_addr_t *
flagcx_uct_ep_addr_ep(const flagcx_uct_ep_addr_t *addr) {
  return (uct_ep_addr_t*)(addr->data + addr->dev_addr_size);
}
 static flagcxResult_t flagcx_uct_ep_addr_set(flagcx_uct_ep_addr_t *addr,
                                          const flagcx_uct_comm_t *comm,
                                          const flagcx_uct_ep_t *uct_ep) {
   flagcx_uct_iface_t *uct_iface = comm->uct_iface;
   size_t total = uct_iface->dev_addr_size + uct_iface->ep_addr_size;
 
   if (total > sizeof(addr->data)) {
          WARN("Address sizes are too big (%zu + %zu > %zu)", uct_iface->dev_addr_size,
               uct_iface->ep_addr_size, sizeof(addr->data));
     return flagcxSystemError;
   }
 
   addr->dev_addr_size = uct_iface->dev_addr_size;
   addr->ep_addr_size  = uct_iface->ep_addr_size;
 
   memcpy(addr->data, uct_iface->dev_addr, addr->dev_addr_size);
   memcpy(addr->data + addr->dev_addr_size, uct_ep->addr,
          uct_iface->ep_addr_size);
  return flagcxSuccess;
}

static flagcxResult_t flagcx_uct_ep_connect_to_ep(flagcx_uct_ep_t *uct_ep,
                                                  flagcx_uct_ep_addr_t *addr) {
  UCXCHECK(uct_ep_connect_to_ep(uct_ep->ep, flagcx_uct_ep_addr_dev(addr),
                                flagcx_uct_ep_addr_ep(addr)),
           return flagcxSystemError, "connect to EP");

  return flagcxSuccess;
}

static flagcxResult_t flagcx_uct_comm_gpu_flush_init(flagcx_uct_comm_t *comm) {
  size_t size = comm->uct_iface->min_get_zcopy;

   comm->gpu_flush.enabled = (flagcx_p2p_gdr_support() == flagcxSuccess) ||
      (flagcx_p2p_dmabuf_support(comm->dev) == flagcxSuccess);

  if (!comm->gpu_flush.enabled) {
    return flagcxSuccess;
  }

   comm->gpu_flush.mem = (uint8_t*)malloc(size);
  if (comm->gpu_flush.mem == NULL) {
    goto fail;
  }

  comm->gpu_flush.uct_ep = flagcx_uct_ep_create(comm->uct_iface);
  if (comm->gpu_flush.uct_ep == NULL) {
    goto fail_free_mem;
  }

  FLAGCXCHECK(flagcx_uct_ep_addr_set(&comm->gpu_flush.addr, comm,
                                     comm->gpu_flush.uct_ep));
   FLAGCXCHECK(
       flagcx_uct_ep_connect_to_ep(comm->gpu_flush.uct_ep, &comm->gpu_flush.addr));
   UCXCHECK(uct_md_mem_reg(comm->uct_iface->md, (void*)comm->gpu_flush.mem, size,
                           UCT_MD_MEM_ACCESS_ALL, &comm->gpu_flush.memh),
           goto fail_destroy_ep, "GPU flush memory registration");

  return flagcxSuccess;

fail_destroy_ep:
  flagcx_uct_ep_destroy(comm->gpu_flush.uct_ep);
fail_free_mem:
  free(comm->gpu_flush.mem);
fail:
  comm->gpu_flush.enabled = 0;
  return flagcxSystemError;
}

flagcxResult_t flagcxUcxGetDevFromName(char *name, int *dev) {
  for (int i = 0; i < flagcxNMergedIbDevs; i++) {
    if (strcmp(flagcxIbMergedDevs[i].devName, name) == 0) {
      *dev = i;
      return flagcxSuccess;
    }
  }
  return flagcxSystemError;
}

static flagcxResult_t flagcx_uct_wr_init() {  
  context.ops.comm_alloc = flagcx_uct_wr_comm_alloc;
  context.ops.comm_init  = flagcx_uct_wr_comm_init;
  context.ops.iface_set  = flagcx_uct_wr_iface_set;
  context.am_short_size  = flagcx_uct_rdesc_size(FLAGCX_UCX_UCT_MAX_RECVS);
  context.rkey_size      = sizeof(((flagcx_uct_chunk_t*)0)->rkey);
  
  flagcxResult_t result = flagcx_p2p_ib_init(&context.dev_count, &context.merge_dev_count, flagcxIbDevs, context.if_name,
                          &context.if_addr, NULL, NULL);  
  return result;
}

flagcxResult_t flagcxUcxInit() {  
  // Initialize IB symbols first since we use flagcxIbDevs
  if (flagcxWrapIbvSymbols() != flagcxSuccess) {
    return flagcxInternalError;
  }
  return flagcx_uct_wr_init();
}
flagcxResult_t flagcxUcxDevices(int *ndev) {
  if (ndev == NULL) {
    return flagcxInvalidUsage;
  }
  
  *ndev = context.dev_count;
  return flagcxSuccess;
}

flagcxResult_t flagcx_p2p_ib_get_properties(flagcxIbDev *devs, int flagcxNMergedIbDevs, int dev, flagcxNetProperties_v8_t* props) {
  if (dev >= flagcxNMergedIbDevs) {
    return flagcxInvalidUsage;
  }
  
  struct flagcxIbMergedDev* mergedDev = flagcxIbMergedDevs + dev;
  struct flagcxIbDev* ibDev = flagcxIbDevs + mergedDev->vProps.devs[0];
  
  props->name = mergedDev->devName;
  props->speed = mergedDev->speed;
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = FLAGCX_PTR_HOST;
  props->regIsGlobal = 1;
  props->latency = 0;
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = ibDev->maxQp;
  props->maxRecvs = FLAGCX_NET_IB_MAX_RECVS;
  props->netDeviceType = FLAGCX_NET_DEVICE_HOST;
  props->netDeviceVersion = FLAGCX_NET_DEVICE_INVALID_VERSION;
  
  return flagcxSuccess;
}
flagcxResult_t flagcxUcxGetProperties(int dev, void *props) {
  flagcxNetProperties_v8_t *properties = (flagcxNetProperties_v8_t *)props;
  return flagcx_p2p_ib_get_properties(flagcxIbDevs, context.merge_dev_count, dev, properties);
}
flagcxResult_t flagcxUcxListen(int dev, void *listen_handle, void **listen_comm) {
  flagcx_uct_listen_handle_t *handle = (flagcx_uct_listen_handle_t*)listen_handle;
  flagcx_uct_listen_comm_t *l_comm   = (flagcx_uct_listen_comm_t*)calloc(1, sizeof(*l_comm));
  flagcx_uct_comm_t *accept_comm;
  union flagcxSocketAddress addr;
  if (l_comm == NULL) {
    WARN("Failed to alloc UCT listener(dev=%d)", dev);
    return flagcxSystemError;
  }
      static_assert(sizeof(flagcx_uct_listen_handle_t) < FLAGCX_NET_HANDLE_MAXSIZE,
                 "UCT listen handle is too big");
  FLAGCXCHECK(flagcxSocketInit(&l_comm->sock, &context.if_addr,
                           FLAGCX_UCT_LISTEN_HANDLE_MAGIC, flagcxSocketTypeNetIb,
                           NULL, 1));
  FLAGCXCHECK(flagcxSocketListen(&l_comm->sock));
  FLAGCXCHECK(flagcxSocketGetAddr(&l_comm->sock, &addr));
  l_comm->uct_worker = flagcx_uct_worker_get(&context, dev);
  if (l_comm->uct_worker == NULL) {
    WARN("Failed to create worker for listener dev=%d", dev);
    return flagcxSystemError;
  }
  FLAGCXCHECK(context.ops.comm_alloc(&accept_comm));
  l_comm->comm    = accept_comm;
  l_comm->context = &context;
  l_comm->dev     = dev;
  l_comm->id      = context.listener_count++;
  *listen_comm = l_comm;
  memset(handle, 0, sizeof(*handle));
  handle->magic         = FLAGCX_UCT_LISTEN_HANDLE_MAGIC;
  handle->listener.id   = l_comm->id;
  handle->listener.addr = addr;
  handle->comm          = accept_comm;
  INFO(FLAGCX_INIT | FLAGCX_NET, "Listening id=%d dev=%d comm=%p", l_comm->id, dev,
       handle->comm);
  return flagcxSuccess;
}

flagcxResult_t flagcxUcxConnect(int dev, void *handle, void **sendComm) {
  //INFO(FLAGCX_NET, "NET/UCX : flagcxUcxConnect called with dev=%d, handle=%p, sendComm=%p", dev, handle, sendComm);
  int ready                        = 0;
  flagcx_uct_listen_handle_t *listen_handle = (flagcx_uct_listen_handle_t*)handle;
  flagcx_uct_stage_t *stage          = &listen_handle->stage;
  flagcx_uct_comm_t *comm            = stage->comm;
  struct {
    flagcx_uct_comm_addr_t       addr;  /* Remote addresses */
    const struct flagcx_uct_comm *comm; /* Cookie received in connect */
  } remote;

  *sendComm = NULL;
  switch (stage->state) {
  case FLAGCX_UCT_START:
    FLAGCXCHECK(context.ops.comm_alloc(&comm));
    FLAGCXCHECK(context.ops.comm_init(comm, &context, NULL, dev, listen_handle->comm));
    FLAGCXCHECK(flagcxSocketInit(&comm->sock, &listen_handle->listener.addr, listen_handle->magic,
                             flagcxSocketTypeNetIb, NULL, 1));
    FLAGCXCHECK(flagcxSocketConnect(&comm->sock));

    stage->comm  = comm;
    stage->state = FLAGCX_UCT_CONNECT;
    /* fallthrough */

  case FLAGCX_UCT_CONNECT:
    FLAGCXCHECK(flagcxSocketReady(&comm->sock, &ready));
    if (!ready) {
      return flagcxSuccess;
    }

    FLAGCXCHECK(flagcx_uct_ep_addr_set(&remote.addr.rma, comm, comm->uct_ep));
    remote.comm = comm;
    FLAGCXCHECK(flagcxSocketSend(&comm->sock, &remote, sizeof(remote)));

    stage->offset = 0;
    stage->state  = FLAGCX_UCT_RECEIVE_ADDR;
    /* fallthrough */

  case FLAGCX_UCT_RECEIVE_ADDR:
    FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_RECV, &comm->sock,
                                 &comm->remote.addr, sizeof(comm->remote.addr),
                                 &stage->offset));
    if (stage->offset != sizeof(comm->remote.addr)) {
      return flagcxSuccess; /* In progress */
    }

    ready = 1;
    FLAGCXCHECK(flagcx_uct_ep_connect_to_ep(comm->uct_ep, &comm->remote.addr.rma));
    FLAGCXCHECK(flagcxSocketSend(&comm->sock, &ready, sizeof(ready)));

    *sendComm   = comm;
    stage->state = FLAGCX_UCT_DONE;
    INFO(FLAGCX_INIT | FLAGCX_NET,
         "Connected comm=%p remote_comm=%p listener_id=%d", comm,
         comm->remote.comm, listen_handle->listener.id);
    break;

  default:
    WARN("UCT connnect for dev=%d using unsupported state %d", dev,
         stage->state);
    return flagcxSystemError;
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxUcxAccept(void *listenComm, void **recvComm) {
  flagcx_uct_listen_comm_t *l_comm = (flagcx_uct_listen_comm_t*)listenComm;
  flagcx_uct_stage_t *stage        = &l_comm->stage;
  flagcx_uct_comm_t *comm          = stage->comm;
  flagcx_uct_comm_addr_t addr;
  int ready;

  *recvComm = NULL;

  switch (stage->state) {
  case FLAGCX_UCT_START:
    comm = l_comm->comm;

    FLAGCXCHECK(flagcxSocketInit(&comm->sock, NULL, FLAGCX_SOCKET_MAGIC,
                             flagcxSocketTypeUnknown, NULL, 0));
    FLAGCXCHECK(flagcxSocketAccept(&comm->sock, &l_comm->sock));
    FLAGCXCHECK(context.ops.comm_init(comm, l_comm->context, l_comm->uct_worker,
                                    l_comm->dev, NULL));
    FLAGCXCHECK(flagcx_uct_comm_gpu_flush_init(comm));

    stage->comm  = comm;
    stage->state = FLAGCX_UCT_ACCEPT;
    /* fallthrough */

  case FLAGCX_UCT_ACCEPT:
    FLAGCXCHECK(flagcxSocketReady(&comm->sock, &ready));
    if (!ready) {
      return flagcxSuccess;
    }

    FLAGCXCHECK(flagcx_uct_ep_addr_set(&addr.rma, comm, comm->uct_ep));
    FLAGCXCHECK(flagcxSocketSend(&comm->sock, &addr, sizeof(addr)));

    stage->offset = 0;
    stage->state  = FLAGCX_UCT_RECEIVE_REMOTE;
    /* fallthrough */

  case FLAGCX_UCT_RECEIVE_REMOTE:
    FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_RECV, &comm->sock, &comm->remote,
                                 sizeof(comm->remote), &stage->offset));
    if (stage->offset != sizeof(comm->remote)) {
      return flagcxSuccess;
    }

    FLAGCXCHECK(flagcx_uct_ep_connect_to_ep(comm->uct_ep, &comm->remote.addr.rma));

    stage->ready  = 0;
    stage->offset = 0;
    stage->state  = FLAGCX_UCT_RX_READY;
    /* fallthrough */

  case FLAGCX_UCT_RX_READY:
    FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_RECV, &comm->sock, &stage->ready,
                                 sizeof(stage->ready), &stage->offset));
    if (stage->offset != sizeof(ready)) {
      return flagcxSuccess;
    }
    if (stage->ready != 1) {
      WARN("Accepted comm=%p invalid status (ready=%d)", comm, stage->ready);
      return flagcxSystemError;
    }

    *recvComm   = comm;
    stage->state = FLAGCX_UCT_DONE;
    INFO(FLAGCX_INIT | FLAGCX_NET, "Accepted comm=%p remote_comm=%p listener_id=%d",
         comm, comm->remote.comm, l_comm->id);
    break;

  default:
    WARN("UCT accept for dev=%d using unsupported state %d", l_comm->dev,
         stage->state);
    return flagcxSystemError;
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxUcxClose(void *close_comm) {
  flagcx_uct_wr_comm_t *comm = flagcx_uct_wr_comm_get((flagcx_uct_comm_t*)close_comm);
  flagcx_uct_rdesc_t *rdesc;

  flagcx_uct_comm_deinit((flagcx_uct_comm_t*)close_comm);

  while ((rdesc = comm->free_rdesc) != NULL) {
    comm->free_rdesc = rdesc->next;
    free(rdesc);
  }

  assert(ucs_list_is_empty(&comm->rdesc_list));
  assert(comm->rdesc_alloc == 0);
  free(comm);
  return flagcxSuccess;
}

flagcxResult_t flagcxUcxCloseListen(void *listen_comm) {
  flagcx_uct_listen_comm_t *comm = (flagcx_uct_listen_comm_t*)listen_comm;

  if (comm) {
    FLAGCXCHECK(flagcxSocketClose(&comm->sock));
    free(comm);
  }
  return flagcxSuccess; 
}

flagcxResult_t flagcxUcxRegMr(void *reg_comm, void *data, size_t size, int type,
                              void **mhandle) {
   flagcx_uct_comm_t *comm = (flagcx_uct_comm_t*)reg_comm;
   uct_component_h comp  = comm->uct_iface->comp;
   uct_md_h md           = comm->uct_iface->md;
   intptr_t addr         = (intptr_t)data;
   size_t rkey_size      = comm->uct_iface->rkey_packed_size;
   flagcx_uct_memh_t *uct_memh;
 
   FLAGCXCHECK(flagcxIbMalloc((void**)&uct_memh, sizeof(*uct_memh) + rkey_size));
   uct_memh->comm = comm;
 
   /* Use integral pages */
   size += addr & (FLAGCX_UCT_REG_ALIGN - 1);
   size  = (size + FLAGCX_UCT_REG_ALIGN - 1) & ~(FLAGCX_UCT_REG_ALIGN - 1);
   addr &= ~(FLAGCX_UCT_REG_ALIGN - 1);
 
   /* Register memory */
   UCXCHECK(uct_md_mem_reg(md, (void*)addr, size, UCT_MD_MEM_ACCESS_ALL,
                           &uct_memh->memh),
            return flagcxSystemError, "register %p/%zu on comm %p", (void*)addr, size,
            (void*)comm);
   /* Pack memory */
   UCXCHECK(uct_md_mkey_pack(md, uct_memh->memh, uct_memh->rkey),
            return flagcxSystemError, "pack rkey for %p/%zu on comm %p", (void*)addr,
            size, (void*)comm);
   /* Unpack rkey from sender side */
   UCXCHECK(uct_rkey_unpack(comp, uct_memh->rkey, &uct_memh->bundle),
            return flagcxInternalError, "unpack rkey");
 
   *mhandle = uct_memh;
   return flagcxSuccess;
}
 
flagcxResult_t flagcxUcxRegMrDmaBuf(void *reg_comm, void *data, size_t size,
                                    int type, uint64_t offset, int fd,
                                    void **mhandle) {
  return flagcxUcxRegMr(reg_comm, data, size, type, mhandle);
}
flagcxResult_t flagcxUcxDeregMr(void *dereg_comm, void *mhandle) {
  flagcx_uct_comm_t *comm     = (flagcx_uct_comm_t*)dereg_comm;
  uct_component_h comp      = comm->uct_iface->comp;
  flagcx_uct_memh_t *uct_memh = (flagcx_uct_memh_t*)mhandle;

  assert(uct_memh->memh != UCT_MEM_HANDLE_NULL);
  assert(uct_memh->comm == comm);

  UCXCHECK(uct_rkey_release(comp, &uct_memh->bundle), , "release rkey bundle");
  UCXCHECK(uct_md_mem_dereg(comm->uct_iface->md, uct_memh->memh),
           return flagcxSystemError, "deregister memh %p on comm %p", uct_memh,
           (void*)comm);

  uct_memh->comm = NULL;
  free(uct_memh);
  return flagcxSuccess;
}


static flagcxResult_t flagcxUcxIsend(void *send_comm, void *data, size_t size,
                                      int tag, void *mhandle,  void* phandle, void **request) {
  flagcx_uct_wr_comm_t *comm = flagcx_uct_wr_comm_get((flagcx_uct_comm_t*)send_comm);
  flagcx_uct_rdesc_t *rdesc;
  int i;

  *request = NULL;

  ucs_list_for_each(rdesc, &comm->rdesc_list, list) {
    for (i = 0; i < rdesc->desc.count; i++) {
      if (rdesc->desc.chunk[i].matched || (rdesc->desc.chunk[i].tag != tag)) {
        continue;
      }

      return flagcx_uct_send(comm, data, size, (flagcx_uct_memh_t*)mhandle, rdesc, i, request);
    }
  }

  /* Progress here to make sure we receive non-solicited RTRs */
  uct_worker_progress(comm->base.uct_worker->worker);
  return flagcxSuccess;
}

static flagcxResult_t flagcxUcxIrecv(void *recv_comm, int n, void **data,
                                      size_t *sizes, int *tags, void **mhandles,
                                      void** phandles, void **request) {
  flagcx_uct_wr_comm_t *comm   = flagcx_uct_wr_comm_get((flagcx_uct_comm_t*)recv_comm);
  flagcx_uct_memh_t **uct_memh = (flagcx_uct_memh_t**)mhandles;
  flagcx_uct_rdesc_t *rdesc;
  ucs_status_t status;

  assert(n <= FLAGCX_UCX_UCT_MAX_RECVS);

  rdesc = flagcx_uct_comm_rdesc_get(comm);
  if (rdesc == NULL) {
    return flagcxInternalError;
  }

  flagcx_uct_rdesc_set(rdesc, comm->rdesc_id++, n, data, sizes, tags, uct_memh);

  status = uct_ep_am_short(comm->base.uct_ep->ep, FLAGCX_UCT_AM_RTR,
                           (uint64_t)comm->base.remote.comm, &rdesc->desc,
                           flagcx_uct_rdesc_size(n));
  if (status != UCS_OK) {
    flagcx_uct_comm_rdesc_put(rdesc);
    *request = NULL;
  } else {
    /* Wait for receiving ATP */
    *request = flagcx_uct_rdesc_get_req(rdesc, 0, FLAGCX_UCT_REQ_IRECV);
  }

  return flagcxSuccess;
}

static flagcxResult_t flagcxUcxIflush(void *recv_comm, int n, void **data,
                                       int *sizes, void **mhandle,
                                       void **request) {
  flagcx_uct_comm_t *base_comm = (flagcx_uct_comm_t*)recv_comm;
  int last                   = flagcx_uct_flush_index(base_comm, sizes, n);
  flagcx_uct_memh_t **uct_memh = (flagcx_uct_memh_t**)mhandle;
  flagcx_uct_rdesc_t *rdesc;
  flagcx_uct_req_t *req;
  flagcxResult_t result;

  if (last == -1) {
    return flagcxSuccess;
  }

  rdesc = flagcx_uct_comm_rdesc_get(flagcx_uct_wr_comm_get(base_comm));
  if (rdesc == NULL) {
    return flagcxInternalError;
  }

  flagcx_uct_rdesc_set(rdesc, ~0, 0, NULL, NULL, NULL, NULL);
  /* Wait for local GET completion */
  req      = flagcx_uct_rdesc_get_req(rdesc, 0, FLAGCX_UCT_REQ_IFLUSH);
  *request = req;

  result = flagcx_uct_flush(base_comm, data[last], sizes[last], uct_memh[last],
                          &req->completion, request);
  if (*request == NULL) {
    flagcx_uct_comm_rdesc_put(rdesc);
  }

  return result;
}

static flagcxResult_t flagcxUcxTest(void *request, int *done, int *sizes) {
  flagcx_uct_req_t *req      = (flagcx_uct_req_t*)request;
  flagcx_uct_rdesc_t *rdesc  = req->rdesc;
  flagcx_uct_wr_comm_t *comm = rdesc->comm;

  uct_worker_progress(comm->base.uct_worker->worker);

  *done = 0;

  if (rdesc->send_atp == 1) {
    /* Slowpath */
    flagcx_uct_send_atp(comm, rdesc);

    if (rdesc->send_atp && rdesc->flagcx_usage == 1) {
      /* Keep the last isend request until ATP is out */
      return flagcxSuccess;
    }
  }

  if (req->completion.count > 0) {
    return flagcxSuccess;
  }

  *done = 1;

  if (req->size == FLAGCX_UCT_REQ_IRECV) {
    assert(&rdesc->reqs[0] == req);
    if (sizes != NULL) {
      memcpy(sizes, rdesc->sizes, rdesc->desc.count * sizeof(*sizes));
    }
  } else if (req->size == FLAGCX_UCT_REQ_IFLUSH) {
    assert(&rdesc->reqs[0] == req);
  } else {
    /* ->isend() request */
    assert(req->size > -1);
    if (sizes != NULL) {
      sizes[0] = req->size;
    }
  }

  if (--rdesc->flagcx_usage < 1) {
    assert(rdesc->send_atp == 0);
    assert(rdesc->flagcx_usage == 0);
    flagcx_uct_comm_rdesc_put(rdesc);
  }

  return flagcxSuccess;
}


// UCX network adaptor structure
struct flagcxNetAdaptor flagcxNetUcx = {
    // Basic functions
    "UCX", flagcxUcxInit, flagcxUcxDevices, flagcxUcxGetProperties, // 111111
    NULL, // reduceSupport
    NULL, // getDeviceMr
    NULL, // irecvConsumed

    // Setup functions
    flagcxUcxListen, // listen 11111
    flagcxUcxConnect, // connect 1111
    flagcxUcxAccept, // accept 11111 
    flagcxUcxClose, // closeSend 1111
    flagcxUcxClose, // closeRecv 1111
    flagcxUcxCloseListen, // closeListen 1111

    // Memory region functions
    flagcxUcxRegMr, // regMr 11111
    flagcxUcxRegMrDmaBuf, // regMrDmaBuf 1111
    flagcxUcxDeregMr, // deregMr 11111

    // Two-sided functions
    flagcxUcxIsend, // isend
    flagcxUcxIrecv, // irecv
    flagcxUcxIflush, // iflush
    flagcxUcxTest, // test

    // One-sided functions
    NULL, // write - TODO: Implement
    NULL, // read - TODO: Implement
    NULL, // signal - TODO: Implement

    // Device name lookup
    flagcxUcxGetDevFromName  // getDevFromName
};


#endif // USE_UCX
