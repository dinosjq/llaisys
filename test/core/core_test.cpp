#include <iostream>

int main() {
    static_assert(1 + 1 == 2, "core test smoke assertion failed");
    std::cout << "llaisys core test smoke assertion passed\n";
    return 0;
}
