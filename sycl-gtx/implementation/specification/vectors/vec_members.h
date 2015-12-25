#pragma once

namespace cl {
namespace sycl {

// Forward declaration
template <typename, int>
class vec;


namespace detail {
namespace vectors {

#define SYCL_SIMPLE_SWIZZLES

// Forward declarations
template <typename, int>
class base;
template <typename dataT, int numElements>
using swizzled_vec = vec<dataT, numElements>;

template <typename dataT, int parentElems, int selfElems = parentElems>
struct members;

template <typename dataT, int parentElems>
struct members<dataT, parentElems, 1> {
#ifndef SYCL_SIMPLE_SWIZZLES
	members(base<dataT, parentElems>* parent) {}
#else
protected:
	base<dataT, parentElems>* parent;

public:
	members(base<dataT, parentElems>* parent)
		: parent(parent) {}

	swizzled_vec<dataT, 1> x() const {
		return this->parent->swizzle<0>();
}
#endif
};

template <typename dataT, int parentElems>
struct members<dataT, parentElems, 2> : members<dataT, parentElems, 1> {
	members(base<dataT, parentElems>* parent)
		: members<dataT, parentElems, 1>(parent) {}

#ifdef SYCL_SIMPLE_SWIZZLES
	swizzled_vec<dataT, 1> y() const {
		return this->parent->swizzle<1>();
	}
#endif
};

template <typename dataT, int parentElems>
struct members<dataT, parentElems, 3> : members<dataT, parentElems, 2> {
	members(base<dataT, parentElems>* parent)
		: members<dataT, parentElems, 2>(parent) {}

#ifdef SYCL_SIMPLE_SWIZZLES
	swizzled_vec<dataT, 1> z() const {
		return this->parent->swizzle<2>();
	}

	swizzled_vec<dataT, 3> xyz() const {
		return this->parent->swizzle<0, 1, 2>();
	}
#endif
};

template <typename dataT, int parentElems>
struct members<dataT, parentElems, 4> : members<dataT, parentElems, 3> {
	members(base<dataT, parentElems>* parent)
		: members<dataT, parentElems, 3>(parent) {}

#ifdef SYCL_SIMPLE_SWIZZLES
	swizzled_vec<dataT, 1> w() const {
		return this->parent->swizzle<3>();
	}
#endif
};

template <typename dataT, int parentElems>
struct members<dataT, parentElems, 8> {
	members(base<dataT, parentElems>* parent) {}

#ifdef SYCL_SIMPLE_SWIZZLES
#endif
};

template <typename dataT, int parentElems>
struct members<dataT, parentElems, 16> {
	members(base<dataT, parentElems>* parent) {}

#ifdef SYCL_SIMPLE_SWIZZLES
#endif
};

} // namespace vectors
} // namespace detail

} // namespace sycl
} // namespace cl
