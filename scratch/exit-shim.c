/* exit-shim.c: interpose exit() and log the native backtrace before dying. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <execinfo.h>
#include <fcntl.h>

static FILE *shim_log = NULL;

static void shim_init(void) __attribute__((constructor));
static void shim_init(void) {
    shim_log = fopen("/tmp/exit_shim.log", "a");
    if (shim_log) {
        setvbuf(shim_log, NULL, _IOLBF, 0);
        fprintf(shim_log, "=== shim active pid=%d ===\n", getpid());
    }
}

void exit(int code) {
    if (shim_log) {
        void *bt[64];
        int n = backtrace(bt, 64);
        fprintf(shim_log, ">>> exit(%d) called, backtrace:\n", code);
        fflush(shim_log);
        backtrace_symbols_fd(bt, n, fileno(shim_log));
        fflush(shim_log);
        /* dump whatever fd2 (stderr) accumulated -- the X error text lives there */
        if (lseek(2, 0, SEEK_CUR) != (off_t)-1) {
            off_t sz = lseek(2, 0, SEEK_END);
            if (sz > 0 && sz < 1048576) {
                char *buf = malloc(sz + 1);
                if (buf) {
                    lseek(2, sz > 4096 ? sz - 4096 : 0, SEEK_SET);
                    ssize_t got = read(2, buf, sz > 4096 ? 4096 : (size_t)sz);
                    if (got > 0) {
                        fprintf(shim_log, ">>> fd2 tail (%zd bytes):\n", got);
                        fwrite(buf, 1, (size_t)got, shim_log);
                        fprintf(shim_log, "\n>>> fd2 dump end\n");
                    }
                    free(buf);
                }
            }
        }
        fflush(shim_log);
    }
    if (shim_log) fclose(shim_log);
    _exit(code);
}
