#include <vector>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <cmath>

using u32 = std::uint32_t;
using u64 = std::uint64_t;

using f32 = float;
using f64 = double;

// conservative variables
struct u_t {
    f64 rho_a;
    f64 rho_ua;
    f64 rho_ea;
};

// primitive variables
struct w_t {
    f64 rho;
    f64 u;
    f64 p;
};

// conservative variables vectors
struct u_vec_t {
    std::vector<f64> rho_a;
    std::vector<f64> rho_ua;
    std::vector<f64> rho_ea;

    void alloc(u32 size) {
        rho_a.resize(size+2);
        rho_ua.resize(size+2);
        rho_ea.resize(size+2);
    }
};

// primitive variables vectors
struct w_vec_t {
    std::vector<f64> rho;
    std::vector<f64> u;
    std::vector<f64> p;
};

struct Mesh {
    std::vector<f64> area;
    const f32 x0;
    const f32 x1;
    f32 dx; 
    u32 n;

    Mesh(u32 size, f32 x0_, f32 x1_): n(size), x0(x0_), x1(x1_) {
        area.resize(size+2);
        dx = (x1 - x0) / (n - 1);
    }

    void generate_grid() {
        std::fill(area.begin(), area.end(), 1.0);
    }

    void generate_nozzle() {
        f32 x = x0;
        for (u32 i = 1; i < n+1; i++) {
            area[i] = 1.398f + 0.347f * std::tanh(0.8f * (x + 0.5*dx) - 4.0f); // nozzle profile 
            x += dx;
        }
        area[0] = area[1]; // left ghost cell 
        area[n+1] = area[n]; // right ghost cell
    }
};

class Scheme {

};

class MacCormack : public Scheme {

};

class BeamWarming : public Scheme {

};


int main(int argc, char** argv) {
    std::cout << "-- HW2 --\n";

    return 0;
}