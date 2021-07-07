#include "../common.h"

// 2.6 Anatomy of a SYCL application
// (slightly revised from revision 2014-09-16 of the specification)

// Originally test8
using namespace cl::sycl;

class my_selector : public device_selector {
 public:
  int operator()(const device& dev) const {
    int score = -1;
    if (dev.is_gpu()) {
      auto vendor = dev.get_info<info::device::vendor>();
      debug() << vendor;

      if (vendor.find("NVIDIA") != std::string::npos) {
        score += 2;
      } else {
        score += 1;
      }
    }
    return score;
  }
};

int main() {
  auto plts = platform::get_platforms();
  for (auto const& plt : plts) {
    for (auto const& dev : plt.get_devices(info::device_type::gpu)) {
      debug() << "Device name: ";
      debug() << dev.get_info<info::device::name>();
    }
  }

  static const int size = 1024;
  int data[size];  // Initialize data to be worked on

  int expected[size];  // Expected results
  for (int i = 0; i < size; ++i) {
    expected[i] = i;
  }

  // By including all the SYCL work in a {} block, we ensure
  // all SYCL tasks must complete before exiting the block
  {
    my_selector nvidia;

    // create a queue to enqueue work to
    queue myQueue(nvidia);

    // wrap our data variable in a buffer
    buffer<int, 1> resultBuf(data, size);

    // create a command_group to issue commands to the queue
    myQueue.submit([&](handler& cgh) {
      // request access to the buffer
      auto writeResult = resultBuf.get_access<access::mode::write>(cgh);

      // enqueue a prallel_for task
      cgh.parallel_for<class simple_test>(range<1>(size), [=](id<1> idx) {
        writeResult[idx] = idx;
      });  // end of the kernel function
    });    // end of our commands for this queue

  }  // end of scope, so we wait for the queued work to complete

  bool success = true;

  // Print result
  for (int i = 0; i < size; ++i) {
    if (data[i] != expected[i]) {
      debug() << i << ",\texpected" << expected[i] << ",\tactual" << data[i];
      success = false;
    }
  }

  return static_cast<int>(!success);
}
