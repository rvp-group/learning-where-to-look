#define N_THREADS 32
#define N_THREADS_CAM 8
#define N_THREADS_ACTIVE_GRID 8 // 3D kernel, max 8 on z

#define SDF_BLOCK_SIZE 8
#define N_THREADS_REDUCE_HASHTABLE 256 // TODO

#define RESOLVE_COLLISION

#ifndef sint
typedef signed int sint;
#endif

#ifndef uint
typedef unsigned int uint;
#endif

#ifndef slong
typedef signed long slong;
#endif

#ifndef ulong
typedef unsigned long ulong;
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef schar
typedef signed char schar;
#endif
