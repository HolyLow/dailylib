#ifndef COMMON_H_
 #define COMMON_H_

class Timer {
public:
  void start() {
    timespec_get(&start_, TIME_UTC);
  }
  void end() {
    timespec_get(&end_, TIME_UTC);
  }
  double duration_ms() {
    time_t d_sec = end_.tv_sec - start_.tv_sec;
    long long int d_nsec = end_.tv_nsec - start_.tv_nsec;
    double duration_ms_ = (double)(d_sec*1E9 + d_nsec) / 1000 / 1000 ;
    return duration_ms_;
  }
private:
  struct timespec start_;
  struct timespec end_;
};

#define DEBUG(format, ...) do {                                               \
  printf("==DEBUG>> <%s : %d : %s>: "format"\n",                              \
      __FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__);                      \
} while(0)

#define DEBUGIF(flag, format, ...) do {                                       \
  if ((flag)) {                                                               \
    printf("==DEBUG>> <%s : %d : %s>: "format"\n",                            \
        __FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__);                    \
  }                                                                           \
} while(0)

#define CHECK(flag, format, ...) do {                                         \
  if (!(flag)) {                                                              \
    printf("==CHECK>> <%s : %d : %s>: "format"\n",                            \
        __FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__);                    \
  }                                                                           \
} while(0)

#define FATAL(flag, format, ...) do {                                         \
  if (!(flag)) {                                                              \
    printf("==FATAL>> <%s : %d : %s>: "format"\n",                            \
        __FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__);                    \
    exit(1);                                                                  \
  }                                                                           \
} while(0)

// #define LOG(format, ...) do {                                                 \
//   fprintf(stdout, "[%s](%d)-<%s>: "format"\n",                                \
//       __FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__);                      \
// } while(0)
//
// #define LOGIF(flag, format, ...) do {                                         \
//   if ((flag)) {                                                               \
//     printf("[%s](%d)-<%s>: "format"\n",                                       \
//         __FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__);                    \
//   }                                                                           \
// } while(0)

#define LOG(format, ...) do {                                                 \
  fprintf(stdout, "---LOG>>> "format"\n",                                     \
      ## __VA_ARGS__);                                                        \
} while(0)

#define LOGIF(flag, format, ...) do {                                         \
  if ((flag)) {                                                               \
    fprintf(stdout, "---LOG>>> "format"\n",                                   \
        ## __VA_ARGS__);                                                      \
  }                                                                           \
} while(0)

#endif
