/* EGL API stub for headless Qt (offscreen platform never issues real GL calls;
 * these satisfy the dynamic linker for libQt6Gui's EGL references). */
typedef void *EGLDisplay; typedef void *EGLSurface; typedef void *EGLContext;
typedef void *EGLConfig; typedef void (*__eglFn)(void); typedef int EGLint; typedef unsigned int EGLBoolean;
static const char __egl_empty[] = "";
EGLBoolean eglChooseConfig(EGLDisplay d, EGLConfig *c, void *m, EGLint n, EGLint *k) { (void)d;(void)c;(void)m;(void)n; if(k)*k=0; return 0; }
EGLContext eglCreateContext(EGLDisplay d, EGLConfig c, EGLContext s, const EGLint *a) { (void)d;(void)c;(void)s;(void)a; return (void*)0; }
EGLSurface eglCreatePbufferSurface(EGLDisplay d, EGLConfig c, const EGLint *a) { (void)d;(void)c;(void)a; return (void*)0; }
EGLBoolean eglDestroyContext(EGLDisplay d, EGLContext c) { (void)d;(void)c; return 1; }
EGLBoolean eglDestroySurface(EGLDisplay d, EGLSurface s) { (void)d;(void)s; return 1; }
EGLBoolean eglGetConfigAttrib(EGLDisplay d, EGLConfig c, EGLint n, EGLint *v) { (void)d;(void)c;(void)n; if(v)*v=0; return 0; }
EGLBoolean eglGetConfigs(EGLDisplay d, EGLConfig *c, EGLint n, EGLint *k) { (void)d;(void)c;(void)n; if(k)*k=0; return 0; }
EGLContext eglGetCurrentContext(void) { return (void*)0; }
EGLDisplay eglGetCurrentDisplay(void) { return (void*)0; }
EGLSurface eglGetCurrentSurface(EGLint r) { (void)r; return (void*)0; }
EGLint eglGetError(void) { return 0x3000; }
__eglFn eglGetProcAddress(const char *p) { (void)p; return (__eglFn)0; }
EGLBoolean eglMakeCurrent(EGLDisplay d, EGLSurface s, EGLSurface sb, EGLContext c) { (void)d;(void)s;(void)sb;(void)c; return 0; }
EGLBoolean eglQueryContext(EGLDisplay d, EGLContext c, EGLint n, EGLint *v) { (void)d;(void)c;(void)n; if(v)*v=0; return 0; }
const char *eglQueryString(EGLDisplay d, EGLint n) { (void)d;(void)n; return __egl_empty; }
EGLBoolean eglSwapBuffers(EGLDisplay d, EGLSurface s) { (void)d;(void)s; return 1; }
EGLBoolean eglSwapInterval(EGLDisplay d, EGLint i) { (void)d;(void)i; return 1; }
EGLBoolean eglInitialize(EGLDisplay d, EGLint *maj, EGLint *min) { (void)d; if(maj)*maj=1; if(min)*min=4; return 0; }
EGLDisplay eglGetDisplay(void *d) { (void)d; return (void*)0; }
EGLBoolean eglBindAPI(unsigned int api) { (void)api; return 1; }
