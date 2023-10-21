#include <vector>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <fstream>
#include <array>

#include "sciplot/sciplot.hpp"

using u32 = std::uint32_t;
using u64 = std::uint64_t;

using f32 = float;
using f64 = double;

// taken from tinyfvm
// row major (M = nb_row, N = nb_col)
template<typename T, size_t M, size_t N>
struct StaticMatrix {
    std::array<T, M*N> buf = {};
    
    int size() const {return N*M;}
    int nb_row() const {return M;}
    int nb_col() const {return N;}

    T (&data())[M][N] {
        return reinterpret_cast<T (&)[M][N]>(*buf.data());
    };

    T& operator()(size_t row, size_t col) {
        return buf[row * N + col];
    };

    StaticMatrix() = default;
    StaticMatrix(const std::initializer_list<std::initializer_list<T>>& init) {
        auto it = buf.begin();
        for (const auto& row : init) {
            for (const auto& elem : row) {
                *it++ = elem;
            }
        }
    }
    ~StaticMatrix() = default;
};

// conservative variables
struct u_t {
    f64 rho_a = 0.0;
    f64 rho_ua = 0.0;
    f64 ea = 0.0;
};

// primitive variables
struct w_t {
    f64 rho = 0.0;
    f64 u = 0.0;
    f64 e = 0.0;
    f64 p = 0.0;
    f64 c = 0.0;
    w_t(const u_t& cons, const f64 a) {
        static const f32 gamma = 1.4;
        rho = cons.rho_a / a;
        u = cons.rho_ua / cons.rho_a;
        e = cons.ea / a;
        p = (gamma - 1.0f) * (cons.ea/a - 0.5 * rho * u * u);
        c = std::sqrt(gamma * (p / rho));
    }

    w_t(const f64 rho_, const f64 u_, const f64 p_) {
        static const f32 gamma = 1.4;
        rho = rho_;
        u = u_;
        p = p_;
        e = p / (gamma - 1) + 0.5 * rho * u*u;
        c = std::sqrt(gamma * (p / rho));
    }
};

// conservative variables vectors
struct u_vec_t {
    std::vector<f64> rho_a;
    std::vector<f64> rho_ua;
    std::vector<f64> ea;

    void alloc(u64 size) {
        rho_a.resize(size);
        rho_ua.resize(size);
        ea.resize(size);
    }

    u_t get(u64 i) const {
        return {rho_a[i], rho_ua[i], ea[i]};
    }

    u64 size() const {
        return rho_a.size();
    }

    void fill(f64 value) {
        std::fill(rho_a.begin(), rho_a.end(), value);
        std::fill(rho_ua.begin(), rho_ua.end(), value);
        std::fill(ea.begin(), ea.end(), value);
    }
};

// primitive variables vectors
struct w_vec_t {
    std::vector<f64> rho;
    std::vector<f64> u;
    std::vector<f64> p;
};

f32 nozzle_A(f32 x) {
    return 1.398f + 0.347f * std::tanh(0.8f * x - 4.0f);
}

f32 nozzle_dA(f32 x) {
    const f32 sech = (1.0f / std::cosh(0.8f * x - 4.0f));
    return 0.347f * sech * sech;
}

struct Mesh {
    std::vector<f64> area; // area
    std::vector<f64> darea; // derivative of area
    f32 x0 = 0.0f;
    f32 x1 = 1.0f;
    u64 n = 100;
    f32 dx = 0.0f; 

    void set_dims(u64 size, f32 x0_, f32 x1_) {
        n = size;
        area.resize(size+2);
        darea.resize(size+2);
        x0 = x0_;
        x1 = x1_;
        dx = (x1 - x0) / (n - 1);
    }

    void generate_grid() {
        std::fill(area.begin(), area.end(), 1.0);
        std::fill(darea.begin(), darea.end(), 0.0);
    }

    void generate_nozzle() {
        for (u64 i = 1; i < n+1; i++) {
            area[i] = nozzle_A((i-1)*dx); // nozzle profile
            darea[i] = nozzle_dA((i-1)*dx); 
        }
        area[0] = area[1]; // left ghost cell 
        darea[0] = darea[1];

        area[n+1] = area[n]; // right ghost cell
        darea[n+1] = darea[n];
    }
};

u_t convective_flux(const w_t& w, const f32 area) {
    u_t f;
    f.rho_a = w.rho * w.u * area;
    f.rho_ua = (w.rho * w.u * w.u + w.p) * area;
    f.ea = w.u * (w.e + w.p) * area;
    return f;
}

class BoundaryCondition {
    public:
    virtual void apply(u_vec_t& u, std::vector<f64>& area) = 0;
    virtual ~BoundaryCondition() = default;
};

class ShockTube : public BoundaryCondition {
    public:
    void apply(u_vec_t& u, std::vector<f64>& area) override {
        const u64 end = u.rho_a.size() - 1;
        u.rho_a[0] = u.rho_a[1];
        u.rho_ua[0] = u.rho_ua[1];
        u.ea[0] = u.ea[1];

        u.rho_a[end] = u.rho_a[end-1];
        u.rho_ua[end] = u.rho_ua[end-1];
        u.ea[end] = u.ea[end-1];
    }
};

class Nozzle : public BoundaryCondition {
    w_t inf; // inlet conditions
    f64 back_pressure_factor = 1.9; // back pressure factor (P_back / P_inlet)
    public:
    void apply(u_vec_t& u, std::vector<f64>& area) override {
        const u64 end = u.rho_a.size() - 1;
        const f64 gamma = 1.4;
        // inlet (assumed always supersonic)
        u.rho_a[0] = inf.rho * area[0];
        u.rho_ua[0] = inf.rho * inf.u * area[0];
        u.ea[0] = inf.e * area[0];

        // outlet
        w_t lc(u.get(end-1), area[end-1]); // last cell

        // supersonic (Blazek Eq 8.20)
        u.rho_a[end] = u.rho_a[end-1];
        u.rho_ua[end] = u.rho_ua[end-1];
        u.ea[end] = u.ea[end-1];

        // subsonic (Blazek Eq 8.23)
        // f64 pb = inf.p * back_pressure_factor;
        // f64 rhob = lc.rho + (pb - lc.p) / (lc.c * lc.c);
        // f64 ub = lc.u + (lc.p - pb) / (lc.rho * lc.c);
        // f64 eb = pb / (gamma - 1) + 0.5 * rhob * ub*ub;
        // u.rho_a[end] = rhob * area[end];
        // u.rho_ua[end] = rhob * ub * area[end];
        // u.ea[end] = eb * area[end];
    }
    Nozzle(f64 rho_, f64 u_, f64 p_): inf(rho_, u_, p_) {};
};

class TimeState {
    public:
    f64 cfl = 1.0;
    f64 residual0 = 1.0; // initial residual
    std::vector<f64> dt;
    virtual void calc_dt(const Mesh& m, const u_vec_t& u) = 0;
    virtual bool has_converged(f64 residual) = 0;
    
    void calc_dt_all(const Mesh& m, const u_vec_t& u) {
        for (u64 i = 1; i < m.n+1; i++) {
            w_t w_c(u.get(i), m.area[i]);
            dt[i] = cfl * m.dx / (std::abs(w_c.u) + w_c.c);
        }
    }

    TimeState(Mesh& m) {
        dt.resize(m.n+2);
    }
    virtual ~TimeState() = default;
};

class Steady : public TimeState {
    public:
    f64 target_convergence;
    void calc_dt(const Mesh& m, const u_vec_t& u) override {
        calc_dt_all(m, u);
    };
    bool has_converged(f64 residual) override {
        return (std::log10(residual0 / residual) > target_convergence);
    }
    Steady(Mesh& m, f64 target_convergence_): TimeState(m), target_convergence(target_convergence_) {};
};

class Transient : public TimeState {
    public:
    f64 t = 0.0;
    f64 t_final = 1.0;
    void calc_dt(const Mesh& m, const u_vec_t& u) override {
        calc_dt_all(m, u);
        f64 min_dt = *std::min_element(dt.begin()+1, dt.end()-1);
        if (t + min_dt > t_final) min_dt = t_final - t;
        t += min_dt;
        std::fill(dt.begin(), dt.end(), min_dt);
    };
    bool has_converged(f64 residual) override {
        return (t == t_final);
    }
    Transient(Mesh& m, f64 t_final_): TimeState(m), t_final(t_final_) {};
};

class Scheme {
    public:
    Mesh& m;
    std::unique_ptr<TimeState> ts;
    std::unique_ptr<BoundaryCondition> bc;

    virtual void solution(u_vec_t& u, u_vec_t& update) = 0;
    Scheme(Mesh& m_): m(m_) {};
    virtual ~Scheme() = default;
};

class MacCormack : public Scheme {
    public:
    u_vec_t pred;
    u_vec_t corr;
    void solution(u_vec_t& u, u_vec_t& update) override {
        bc->apply(u, m.area);
        // predictor
        for (u64 i = 1; i < m.n+1; i++) {
            w_t w_c = w_t(u.get(i), m.area[i]); // w cell
            w_t w_r = w_t(u.get(i+1), m.area[i+1]); // w left cell

            u_t f_c = convective_flux(w_c, m.area[i]);
            u_t f_r = convective_flux(w_r, m.area[i+1]);
            
            pred.rho_a[i] = u.rho_a[i] - ts->dt[i] * (f_r.rho_a - f_c.rho_a) / m.dx;
            pred.rho_ua[i] = u.rho_ua[i] - ts->dt[i] * (f_r.rho_ua - f_c.rho_ua) / m.dx + ts->dt[i] * w_c.p * m.darea[i];
            pred.ea[i] = u.ea[i] - ts->dt[i] * (f_r.ea - f_c.ea) / m.dx;
        }

        bc->apply(pred, m.area);

        // corrector
        for (u64 i = 1; i < m.n+1; i++) {
            w_t w_c = w_t(pred.get(i), m.area[i]); // w cell
            w_t w_l = w_t(pred.get(i-1), m.area[i-1]); // w right cell

            u_t f_c = convective_flux(w_c, m.area[i]);
            u_t f_l = convective_flux(w_l, m.area[i-1]);

            corr.rho_a[i] = u.rho_a[i] - ts->dt[i] * (f_c.rho_a - f_l.rho_a) / m.dx;
            corr.rho_ua[i] = u.rho_ua[i] - ts->dt[i] * (f_c.rho_ua - f_l.rho_ua) / m.dx + ts->dt[i] * w_c.p * m.darea[i];
            corr.ea[i] = u.ea[i] - ts->dt[i] * (f_c.ea - f_l.ea) / m.dx;
        }

        // update
        for (u64 i = 1; i < m.n+1; i++) {
            update.rho_a[i] = 0.5 * (pred.rho_a[i] + corr.rho_a[i]) - u.rho_a[i];
            update.rho_ua[i] = 0.5 * (pred.rho_ua[i] + corr.rho_ua[i]) - u.rho_ua[i];
            update.ea[i] = 0.5 * (pred.ea[i] + corr.ea[i]) - u.ea[i];
        }
    };

    MacCormack(Mesh& m_): Scheme(m_) {
        pred.alloc(m.n+2);
        corr.alloc(m.n+2);
    }
};

using block_t = StaticMatrix<f64,3,3>;
using jacobian_t = std::vector<block_t>;

void compute_convective_jacobian(u_vec_t& u, std::vector<f64>& area, jacobian_t& jac) {
    static const f64 gamma = 1.4;
    for (u64 i = 0; i < u.size(); i++) {
        w_t c(u.get(i), area[i]); // cell primitives
        jac[i](0,0) = 0.0;
        jac[i](0,1) = 1.0;
        jac[i](0,2) = 0.0;

        jac[i](1,0) = (gamma - 3.0) * 0.5 * c.u * c.u;
        jac[i](1,1) = (3.0 - gamma) * c.u;
        jac[i](1,2) = gamma - 1.0;

        jac[i](2,0) = - gamma * c.e * c.u / c.rho + (gamma - 1.0) * c.u * c.u * c.u;
        jac[i](2,1) = gamma * c.e / c.rho - 1.5 * (gamma - 1.0) * c.u * c.u;
        jac[i](2,2) = gamma * c.u;        
    }
}

void compute_source_jacobian(u_vec_t& u, std::vector<f64>& area, std::vector<f64>& darea, jacobian_t& jac) {
    static const f64 gamma = 1.4;
    for (u64 i = 0; i < u.size(); i++) {
        w_t c(u.get(i), area[i]); // cell primitives
        jac[i](1,0) = 0.5 * (gamma - 1.0) * (darea[i] / area[i]) * c.u * c.u;
        jac[i](1,1) = (1.0 - gamma) * (darea[i] / area[i]) * c.u;
        jac[i](1,2) = (gamma - 1.0) * (darea[i] / area[i]);
    }
}

// replace this with Eigen ffs
// matrix-vector product with matrix in row major
template<size_t N>
void matvec(double A[N][N], double B[N], double C[N]) {
    for (size_t i = 0; i < N; i++) {
        C[i] = 0.0;
    }
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            C[i] += A[i][j] * B[j]; // Multiply and accumulate
        }
    }
}

// a * A + b (only diag)
template<size_t N>
void matscale(double A[N][N], double a, double b) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i][j] *= a;
            //A[i][j] += b;
        }
        A[i][i] += b;
    }
}

template<size_t N>
void matadd(double A[N][N], double B[N][N]) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i][j] += B[i][j];
        }
    }
}

// imported from tinyfvm
template<size_t N>
void small_inv(double A[N][N]) {
    double L[N][N] = {};
    double U[N][N] = {};
    double aux[N] = {};
    double dot;
    std::memcpy(U, A, sizeof(U));

    // LU decomposition
    // L's diag has zeros and U's diag are inverted
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            double factor = U[j][i] / U[i][i];
            L[j][i] = factor;
            for (int k = 0; k < N; k++) {
                U[j][k] -= factor * U[i][k];
            }
        }
        U[i][i] = 1.0 / U[i][i]; // precompute the inverse of diag
    }

    for (int i = 0; i < N; i++) {
        // forward substitution
        for (int j = 0; j < i; j++) {
            dot = 0.0;
            for (int k = 0; k < j; k++) {
                dot += L[j][k] * aux[k];
            }
            aux[j] = - dot;
        }
        dot = 0.0;
        for (int k = 0; k < i; k++) {
            dot += L[i][k] * aux[k];
        }
        aux[i] = 1.0 - dot;
        for (int j = i+1; j < N; j++) {
            dot = 0.0;
            for (int k = 0; k < j; k++) {
                dot += L[j][k] * aux[k];
            }
            aux[j] = - dot;
        }

        // backward substitution
        for (int j = N; j != 0;) {
            j--;
            dot = 0.0;
            for (int k = j+1; k < N; k++) {
                dot += U[j][k] * aux[k];
            }
            aux[j] = (aux[j] - dot) * U[j][j]; // U diag already inverted
        }

        // modify single column
        for (int j = 0; j < N; j++) {
            A[j][i] = aux[j]; // scatter
        }
    }
}

class BeamWarming : public Scheme {
    public:
    jacobian_t jac; // convective jacobian
    jacobian_t jac_s; // source jacobian
    u_vec_t rhs; // precomputed convective flux
    const f64 eps_e = 0.125;
    const f64 eps_i = 2.5 * eps_e;

    void compute_rhs(u_vec_t& u) {
        for (u64 i = 1; i < u.size()-1; i++) {
            u_t u_l = u.get(i-1);
            u_t u_c = u.get(i);
            u_t u_r = u.get(i+1);

            w_t w_l(u_l, m.area[i-1]);
            w_t w_c(u_c, m.area[i]);
            w_t w_r(u_r, m.area[i+1]);

            // yes im computing same flux multiple times but whatever
            u_t f_l = convective_flux(w_l, m.area[i-1]);
            //u_t f_c = convective_flux(w_c, m.area[i]);
            u_t f_r = convective_flux(w_r, m.area[i+1]);

            u_t dissip;

            if (i == 1 || i == u.size()-2) {
                dissip.rho_a = eps_e * (u_r.rho_a - 2.0 * u_c.rho_a + u_l.rho_a);
                dissip.rho_ua = eps_e * (u_r.rho_ua - 2.0 * u_c.rho_ua + u_l.rho_ua);
                dissip.ea = eps_e * (u_r.ea - 2.0 * u_c.ea + u_l.ea);
            } else {
                u_t u_rr = u.get(i+2);
                u_t u_ll = u.get(i-2);
                dissip.rho_a = eps_e * (u_rr.rho_a - 4.0 * u_r.rho_a + 6.0 * u_c.rho_a - 4.0 * u_l.rho_a + u_ll.rho_a);
                dissip.rho_ua = eps_e * (u_rr.rho_ua - 4.0 * u_r.rho_ua + 6.0 * u_c.rho_ua - 4.0 * u_l.rho_ua + u_ll.rho_ua);
                dissip.ea = eps_e * (u_rr.ea - 4.0 * u_r.ea + 6.0 * u_c.ea - 4.0 * u_l.ea + u_ll.ea);
            }
            
            rhs.rho_a[i] = - ts->dt[i] * (f_r.rho_a - f_l.rho_a) / (2.0 * m.dx) - dissip.rho_a;
            rhs.rho_ua[i] = - ts->dt[i] * (f_r.rho_ua - f_l.rho_ua) / (2.0 * m.dx) + ts->dt[i] * w_c.p * m.darea[i] - dissip.rho_ua;
            rhs.ea[i] = - ts->dt[i] * (f_r.ea - f_l.ea) / (2.0 * m.dx) - dissip.ea;
        }
    }

    f64 gs_sweep(u_vec_t& u, u_vec_t& update) {
        f64 residual = 0.0;

        for (u64 i = 1; i < u.size() - 1; i++) {
            u_t acc = rhs.get(i);
            block_t b_r = jac[i+1];
            block_t b_l = jac[i-1];
            block_t diag = jac_s[i];
            f64 upd_r[3] = {update.rho_a[i+1], update.rho_ua[i+1], update.ea[i+1]};
            f64 upd_l[3] = {update.rho_a[i-1], update.rho_ua[i-1], update.ea[i-1]};
            f64 c_r[3] = {};
            f64 c_l[3] = {};
            f64 C[3] = {};
        
            matscale<3>(b_r.data(), ts->dt[i] / (4.0 * m.dx), -eps_i * m.area[i] / m.area[i+1]);
            matscale<3>(b_l.data(), - ts->dt[i] / (4.0 * m.dx), -eps_i * m.area[i] / m.area[i-1]);
            matscale<3>(diag.data(), - 0.5f * ts->dt[i], 1.0f + 2.0f * eps_i);

            matvec<3>(b_r.data(), upd_r, c_r);
            matvec<3>(b_l.data(), upd_l, c_l);

            acc.rho_a -= (c_r[0] + c_l[0]);
            acc.rho_ua -= (c_r[1] + c_l[1]);
            acc.ea -= (c_r[2] + c_l[2]);

            f64 B[3] = {acc.rho_a, acc.rho_ua, acc.ea};

            small_inv<3>(diag.data());
            matvec<3>(diag.data(), B, C);

            // multiply by inv of diag block but rn is just I
            f64 diff = C[0] - update.rho_a[i];
            residual += diff * diff;

            update.rho_a[i] = C[0];
            update.rho_ua[i] = C[1];
            update.ea[i] = C[2];
        }
        return std::sqrt(residual) / (u.size() - 2); // residual normalizating
    };

    void solution(u_vec_t& u, u_vec_t& update) override {
        bc->apply(u, m.area); // applied before anything
        compute_rhs(u);
        compute_convective_jacobian(u, m.area, jac);
        compute_source_jacobian(u, m.area, m.darea, jac_s);

        update.fill(0.0);

        f64 gs_residual0 = gs_sweep(u, update);
        f64 gs_residual = gs_residual0;
        u32 sweep = 1;
        while(std::log10(gs_residual0 / gs_residual) < 5.0 && sweep < 200) {
            //std::cout << sweep << " | GS res: " << gs_residual << "\n";
            gs_residual = gs_sweep(u, update);
            sweep++;
        }
    }

    BeamWarming(Mesh& m_): Scheme(m_) {
        jac.resize(m.n+2); // using all
        jac_s.resize(m.n+2); // using all
        rhs.alloc(m.n+2); // but only using [1:end-1]
    }
};

f64 update_solution(u_vec_t& u, u_vec_t& update) {
    f64 residual_l2 = 0.0;
    for (u64 i = 1; i < u.size()-1; i++) {
        u.rho_a[i] += update.rho_a[i];
        u.rho_ua[i] += update.rho_ua[i];
        u.ea[i] += update.ea[i];
        residual_l2 += update.rho_a[i] * update.rho_a[i];
    }
    return std::sqrt(residual_l2) / (u.rho_a.size() - 2);
}

int main(int argc, char** argv) {
    std::cout << "-- HW2 --\n";

    const std::vector<std::string> schemes = {"mac-cormack", "beam-warming"};
    const std::vector<std::string> test_cases = {"shock-tube", "nozzle"};
    
    u32 scheme_idx = 1;
    u32 test_case_idx = 1;
    u64 n = 200;

    std::string scheme = schemes.at(scheme_idx);
    std::string test_case = test_cases.at(test_case_idx);

    Mesh mesh;
    std::unique_ptr<Scheme> s;

    u_vec_t u; // u_n
    u_vec_t update; // delta_u_n
    u.alloc(n+2);
    update.alloc(n+2);

    // Setup test case
    if (test_case == "shock-tube") {
        mesh.set_dims(n, 0.0f, 1.0f);
        mesh.generate_grid();
        // w_t left(4.0, 0.0, 4.0);
        // w_t right(1.0, 0.0, 1.0);
        w_t left(1.0, 0.0, 1.0);
        w_t right(0.125, 0.0, 0.1);
        f32 x = 0.0f;
        for (u64 i = 1; i < n+1; i++) {
            if (x < 0.5f) {
                u.rho_a[i] = left.rho * mesh.area[i];
                u.rho_ua[i] = left.rho * left.u * mesh.area[i];
                u.ea[i] = left.e * mesh.area[i];
            } else {
                u.rho_a[i] = right.rho * mesh.area[i];
                u.rho_ua[i] = right.rho * right.u * mesh.area[i];
                u.ea[i] = right.e * mesh.area[i];
            }
            x += mesh.dx;
        }
    } else if (test_case == "nozzle") {
        mesh.set_dims(n, 0.0f, 10.0f);
        mesh.generate_nozzle();
        f64 u_vel = 1.3 * std::sqrt(1.4); // we set p = rho = 1.0 and mach = 1.3
        w_t inf(1.0, u_vel, 1.0);
        for (u64 i = 1; i < n+1; i++) {
            u.rho_a[i] = inf.rho * mesh.area[i];
            u.rho_ua[i] = inf.rho * inf.u * mesh.area[i];
            u.ea[i] = inf.e * mesh.area[i];
        }
    } else {
        std::cout << "Test case not implemented\n";
        return 1;
    }

    // Setup scheme
    if (scheme == "mac-cormack") {
        s = std::make_unique<MacCormack>(mesh);
    } else if (scheme == "beam-warming") {
        s = std::make_unique<BeamWarming>(mesh);
    } else {
        std::cout << "Scheme not implemented\n";
        return 1;
    }

    if (test_case == "shock-tube") {
        s->bc = std::make_unique<ShockTube>();
        s->ts = std::make_unique<Transient>(mesh, 0.2); // t_final = 250
    } else if (test_case == "nozzle") {
        s->bc = std::make_unique<Nozzle>(1.0, 1.3 * std::sqrt(1.4), 1.0);
        s->ts = std::make_unique<Steady>(mesh, 10); // converge to 1e-10
    }
    
    // First iteration
    u64 iterations = 0;
    s->ts->calc_dt(mesh, u);
    s->solution(u, update);
    f64 residual0 = update_solution(u, update);
    s->ts->residual0 = residual0;
    f64 l2res = residual0;

    // Solver loop
    while (!s->ts->has_converged(l2res)) {
        s->ts->calc_dt(mesh, u);
        s->solution(u, update);
        l2res = update_solution(u, update);
        if (iterations % 10 == 0) std::cout << iterations << " | Residual: " << std::log10(l2res / residual0) << "\n";
        iterations++;
    }

    std::cout << "Iterations: " << iterations << "\n";

    if (test_case == "shock-tube") {
        std::ifstream sod_tube("sod-exact_solution.txt");
        std::vector<f64> x_exact;
        std::vector<f64> rho_exact;
        while (!sod_tube.eof()) {
            f64 x, rho, u, p;
            sod_tube >> x >> rho >> u >> p;
            x_exact.push_back(x);
            rho_exact.push_back(rho);
        }

        sciplot::Vec x = sciplot::linspace(0.0, 1.0, n+2);
        sciplot::Plot2D plot1;
        plot1.drawCurve(x, u.rho_a).label(scheme).lineWidth(2);
        plot1.drawCurve(x_exact, rho_exact).label("exact").lineWidth(2);

        sciplot::Figure figure = {{plot1}};
        sciplot::Canvas canvas = {{figure}};
        canvas.size(1280, 720);
        canvas.save("sod_tube.png");
    } else if (test_case == "nozzle") {
        std::vector<f64> mach(n+2);
        for (u64 i = 1; i < n+1; i++) {
            w_t w_c(u.get(i), mesh.area[i]);
            mach[i] = w_c.u / w_c.c;
        }
        mach[0] = mach[1];
        mach[n+1] = mach[n];

        sciplot::Vec x = sciplot::linspace(0.0, 10.0, n+2);
        sciplot::Plot2D plot1;
        plot1.drawCurve(x, mach).label("mach").lineWidth(2);

        sciplot::Figure figure = {{plot1}};
        sciplot::Canvas canvas = {{figure}};
        canvas.size(1280, 720);
        canvas.save("nozzle_mach.png");
    }

    return 0;
}