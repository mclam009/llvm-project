// clang-format off
// RUN: %libomp-cxx-compile -pthread && env KMP_GTID_MODE=1 %libomp-run
// REQUIRES: ompt
// clang-format on

// When a root thread that the runtime did not create, such as a std::thread,
// exits, the callback for the end of its initial task
// (ompt_callback_implicit_task, ompt_scope_end, ompt_task_initial) must run in
// the context of that task: the tool gets the task_data that the task began
// with, and ompt_get_thread_data and ompt_get_task_info find the thread and
// the task, as they do in the callbacks of the begin of the task.
//
// The runtime unregisters the thread from the destructor of its thread-specific
// data, which POSIX calls after it cleared the thread-specific value. Where the
// gtid is found by the address of the stack (KMP_GTID_MODE=1, the default on
// macOS) a root thread's stack is only known from the addresses where the
// runtime has seen the thread, and the destructor runs above them when the
// thread's OpenMP calls were deeper in its stack than the destructor is. The
// thread was then not found, and the tool was given uninitialized pointers.
//
// This program is the tool: it records the task_data that the initial task of
// the thread begins with, and checks in the callback of its end that it is
// given the same one and that the inquiry functions find the thread and that
// task. The thread makes its OpenMP calls 64 KiB down its stack.

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <thread>

#include <omp-tools.h>
#include <omp.h>
#include <pthread.h>

namespace {
// The thread under test, which the callbacks look for, and what they saw of it.
// The callbacks take no lock: they also run for the main thread, as the runtime
// shuts down, when static objects have been destroyed.
std::atomic<bool> worker_set{false};
pthread_t worker;
std::atomic<int> begins{0}, ends{0};
std::atomic<ompt_data_t *> begin_task{nullptr};
std::atomic<bool> same_task{false}, thread_found{false}, task_found{false};

ompt_get_thread_data_t get_thread_data = nullptr;
ompt_get_task_info_t get_task_info = nullptr;

void implicit_task(ompt_scope_endpoint_t endpoint, ompt_data_t *parallel,
                   ompt_data_t *task, unsigned, unsigned, int flags) {
  if (!(flags & ompt_task_initial) || !worker_set.load() ||
      !pthread_equal(pthread_self(), worker))
    return;
  if (endpoint == ompt_scope_begin) {
    ++begins;
    begin_task = task;
  } else if (endpoint == ompt_scope_end) {
    ++ends;
    same_task = task == begin_task.load();
    thread_found = get_thread_data() != nullptr;
    ompt_data_t *current = nullptr;
    // 2: a task, with a task_data, was found
    const int status =
        get_task_info(0, nullptr, &current, nullptr, nullptr, nullptr);
    task_found = status == 2 && current == task;
  }
}

int initialize(ompt_function_lookup_t lookup, int, ompt_data_t *) {
  const auto set_callback =
      reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
  get_thread_data =
      reinterpret_cast<ompt_get_thread_data_t>(lookup("ompt_get_thread_data"));
  get_task_info =
      reinterpret_cast<ompt_get_task_info_t>(lookup("ompt_get_task_info"));
  set_callback(ompt_callback_implicit_task,
               reinterpret_cast<ompt_callback_t>(implicit_task));
  return 1;
}

void finalize(ompt_data_t *) {}

[[gnu::noinline]] void open_regions(int &sum) {
  for (int r = 0; r < 2; ++r) {
#pragma omp parallel reduction(+ : sum)
    sum += 1;
  }
}

// Runs the regions 64 KiB down the stack, which the optimiser cannot remove, as
// it can a local array.
[[gnu::noinline]] void deep(int &sum, std::size_t depth) {
  char *const pad = static_cast<char *>(__builtin_alloca(depth));
  asm volatile("" : : "r"(pad) : "memory");
  open_regions(sum);
  asm volatile("" : : "r"(pad) : "memory");
}
} // namespace

extern "C" ompt_start_tool_result_t *ompt_start_tool(unsigned, const char *) {
  static ompt_start_tool_result_t result = {initialize, finalize, {0}};
  return &result;
}

int main() {
  int sum = 0;
  std::thread thread([&] {
    worker = pthread_self();
    worker_set = true;
    deep(sum, 65536);
  });
  thread.join();
  std::printf("the initial task of a std::thread began %d time(s) and ended "
              "%d time(s)\n",
              begins.load(), ends.load());
  if (begins != 1 || ends != 1) {
    std::printf("FAIL: the tool did not get one begin and one end of the "
                "initial task\n");
    return 1;
  }
  std::printf("at its end: task_data %s its begin's, ompt_get_thread_data %s, "
              "ompt_get_task_info %s it\n",
              same_task ? "is" : "is NOT",
              thread_found ? "finds the thread" : "finds NO thread",
              task_found ? "finds" : "does NOT find");
  if (!(same_task && thread_found && task_found)) {
    std::printf("FAIL\n");
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
