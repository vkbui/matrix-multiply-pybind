#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern void cu_matrixMultiply(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols);

namespace py = pybind11;


py::array_t<float> mm_wrapper(py::array_t<float> a1, py::array_t<float> a2)
{
	auto buf1 = a1.request();
	auto buf2 = a2.request();

	if (a1.ndim() != 2 || a2.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");

	if (a1.shape()[1] != a2.shape()[0])
		throw std::runtime_error("Input shapes must match");

	// NxM matrix
	int A_rows = a1.shape()[0];
	int A_cols = a1.shape()[1];
	int B_cols = a2.shape()[1];
	//printf("A_rows=%d, A_cols=%d\n, B_cols=%d", A_rows, A_cols, B_cols);

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ A_rows, B_cols },  /* Number of elements for each dimension */
		{ sizeof(float) * B_cols, sizeof(float) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();

	float* A = (float*)buf1.ptr;
	float* B = (float*)buf2.ptr;
	float* C = (float*)buf3.ptr;

	cu_matrixMultiply(A, B, C, A_rows, A_cols, B_cols);

	return result;
}



PYBIND11_MODULE(cu_matrix_multiply, m) {
	m.def("mm", &mm_wrapper, "Multiply two NumPy arrays");

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}
