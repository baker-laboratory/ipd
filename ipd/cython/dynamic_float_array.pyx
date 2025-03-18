cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
import numpy as np

cdef class DynamicFloatArray:
    cdef float* data
    cdef int size
    cdef int capacity

    def __cinit__(self, int initial_capacity=10):
        """Initialize a resizable dynamic array of floats."""
        self.capacity = initial_capacity
        self.size = 0
        self.data = <float*> malloc(self.capacity * sizeof(float))
        if not self.data:
            raise MemoryError("Failed to allocate memory for float array")

    def append(self, float value):
        """Append a float value, resizing if necessary."""
        if self.size >= self.capacity:
            self._resize()
        self.data[self.size] = value
        self.size += 1

    cdef void _resize(self):
        """Double the capacity of the array when resizing."""
        cdef int new_capacity = self.capacity * 2
        cdef float* new_data = <float*> malloc(new_capacity * sizeof(float))
        if not new_data:
            raise MemoryError("Failed to allocate memory for resizing")
        memcpy(new_data, self.data, self.size * sizeof(float))
        free(self.data)
        self.data = new_data
        self.capacity = new_capacity

    def get_data(self):
        """Return stored values as a NumPy array."""
        return np.asarray(<float[:self.size]> self.data, dtype=np.float32)

    def get_size(self):
        """Return the current number of stored values."""
        return self.size

    def clear(self):
        """Reset the array to an empty state without reallocating memory."""
        self.size = 0

    def __getitem__(self, int index):
        """Allow Python-style indexing."""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")
        return self.data[index]

    def __setitem__(self, int index, float value):
        """Allow setting values using Python-style indexing."""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")
        self.data[index] = value

    def extend(self, list values):
        """Extend the array with multiple float values."""
        for value in values:
            self.append(value)

    def sum(self):
        """Return the sum of stored values."""
        cdef float total = 0
        for i in range(self.size):
            total += self.data[i]
        return total

    def mean(self):
        """Return the mean of stored values."""
        if self.size == 0:
            raise ValueError("Cannot compute mean of empty array")
        return self.sum() / self.size

    def min(self):
        """Return the minimum stored value."""
        if self.size == 0:
            raise ValueError("Cannot compute min of empty array")
        cdef float min_val = self.data[0]
        for i in range(1, self.size):
            if self.data[i] < min_val:
                min_val = self.data[i]
        return min_val

    def max(self):
        """Return the maximum stored value."""
        if self.size == 0:
            raise ValueError("Cannot compute max of empty array")
        cdef float max_val = self.data[0]
        for i in range(1, self.size):
            if self.data[i] > max_val:
                max_val = self.data[i]
        return max_val

    def __len__(self):
        """Return the current size of the array for Python len() compatibility."""
        return self.size

    def __dealloc__(self):
        """Free allocated memory when the object is deleted."""
        if self.data:
            free(self.data)
