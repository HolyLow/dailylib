#ifndef __TIMER_H__
 #define __TIMER_H__

#include <time.h>

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
    double duration_ms_ = 
        (double)(d_sec*1E9 + d_nsec) / 1000 / 1000;
    return duration_ms_;
  }
private:
  struct timespec start_;
  struct timespec end_;
};


#endif
