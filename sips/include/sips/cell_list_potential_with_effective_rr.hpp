#ifndef SIPS_CELL_LIST_POTENTIAL_WITH_EFFECTIVE_RR_HPP
#define SIPS_CELL_LIST_POTENTIAL_WITH_EFFECTIVE_RR_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <omp.h>
#include <math.h>
#include <random>
#include <vector>

#include "hyperalg/distance.hpp"
#include "hyperalg/cell_list_potential.hpp"
#include "hyperalg/vecN.hpp"
#include "utils/noise.hpp"

namespace ha{

// effective SDE (sum of time-delayed white noises) of RR-SGD
// noise: \theta = \langle\theta\rangle + \zeta = b_f + \zeta
// equation of motion: \Delta x_{i} = -\alpha (b_f + \zeta_{ji}) \nabla_{i} V_{ji}
    // multiplicative + time-delayed noise

inline double effective_rr_temporal_c(int K, int ell){
    // Time correlation in Eq. (5)
    // c_0 = 1
    // if 1 <= l <= (K - 1), c_l = -(K-ell)/(K(K-1)).
    // if l >= K, c_l = 0.
    if(ell == 0){
        return 1.0;
    }
    if(1 <= ell && ell <= K - 1){
        return -double(K - ell) / double(K * (K - 1));
    }
    return 0.0;
}

inline std::vector<double> effective_rr_ma_coefficients(int K, double bf){
    // Moving average coefficients
        // check R(l) = b_f (1 - b_f) c_l and \sum_{l}{c_l} = ((-1)*(1 / 2))
    const double scale = std::sqrt(bf / double(K));
    std::vector<double> A(size_t(K) * size_t(K), 0.0);
    for(int stream = 0; stream < K; ++stream){
        for(int lag = 0; lag < K; ++lag){
            const double delta = (lag == stream) ? 1.0 : 0.0;
            A[size_t(stream) * size_t(K) + size_t(lag)] = scale * (delta - 1.0 / double(K));
        }
    }
    const double variance = bf * (1.0 - bf);
    for(int ell = 0; ell < K; ++ell){
        double acc = 0.0;
        for(int stream = 0; stream < K; ++stream){
            for(int lag = 0; lag < K - ell; ++lag){
                const double left = A[size_t(stream) * size_t(K) + size_t(lag)];
                const double right = A[size_t(stream) * size_t(K) + size_t(lag + ell)];
                acc += left * right;
            }
        }
        const double target = variance * effective_rr_temporal_c(K, ell);
        if(std::abs(acc - target) > 1e-10){
            throw std::runtime_error("effective RR: MA coefficients do not match Eq. (5)");
        }
            // check R(l) = b_f (1 - b_f) c_l
    }
    if(K >= 2){
        double total = 0.0;
        for(int ell = 1; ell < K; ++ell){
            total += effective_rr_temporal_c(K, ell);
        }
        if(std::abs(total + 0.5) > 1e-12){
            throw std::runtime_error("effective RR: Eq. (6) requires sum of c_ell = -1/2");
        }
            // check \sum_{l}{c_l} = ((-1)*(1 / 2))
    }
    return A;
}

template <typename pairwise_interaction, typename distance_policy>
class EffectiveRREnergyGradientAccumulator{
protected:
    const static size_t m_ndim = distance_policy::_ndim;
    struct PairHistory{
        std::vector<double> buf;
        long long last;
    };
    struct Stripe{
        std::mutex mutex;
        std::unordered_map<std::uint64_t, PairHistory> pairs;
    };
    struct ThreadNoise{
        std::mt19937_64 generator;
        std::normal_distribution<double> normal;
        explicit ThreadNoise(std::uint64_t seed)
            : generator(seed), normal(0.0, 1.0) {}
        double operator()(){ return normal(generator); }
    };

    std::shared_ptr<pairwise_interaction> m_interaction;
    std::shared_ptr<distance_policy> m_dist;
    const std::vector<double>* m_coords;
    std::vector<double> m_radii;
    double m_alpha;
    double m_bf;
    double m_c;
    double m_rho;
    int m_K;
    std::vector<double> m_A;
    long long m_step;
    size_t m_recorded_pairs;
    bool m_record_selection;
    std::vector<double> m_selection_trace;
    std::vector<std::vector<std::uint64_t>> m_thread_selection_key;
    std::vector<std::vector<double>> m_thread_selection_zeta;
    std::vector<std::unique_ptr<Stripe>> m_stripes;
    std::vector<ThreadNoise> m_noises;
    std::vector<std::vector<double>> m_thread_grad;
    std::vector<std::vector<double>> m_thread_pair_sum;
    std::vector<double> m_thread_energy;
    size_t m_natoms;

    static void two_normals(ThreadNoise& rng, double rho, double& a, double& b){
        // spatial pair-correlation
        // See CorrelatedGaussianNoise::rand2
        // c = (-1) * rho
        a = rng();
        b = rng();
        const double factor = std::sqrt(std::max(0.0, 1.0 - rho * rho));
        b = rho * a + factor * b;
    }

    void fill_age(PairHistory& hist, int age, ThreadNoise& rng){
        for(int stream = 0; stream < m_K; ++stream){
            double a = 0.0;
            double b = 0.0;
            two_normals(rng, m_rho, a, b);
            const size_t base = (size_t(age) * size_t(m_K) + size_t(stream)) * 2;
            hist.buf[base] = a;
            hist.buf[base + 1] = b;
        }
    }

    void refresh(PairHistory& hist, ThreadNoise& rng){
        // keep pair MA
        const long long step = m_step;
        const int K = m_K;
        if(hist.last < 0 || step - hist.last >= K){
            // c_l = 0 for l >= K
            hist.buf.assign(size_t(K) * size_t(K) * 2, 0.0);
            for(int age = K - 1; age >= 0; --age){
                fill_age(hist, age, rng);
            }
            hist.last = step;
            return;
        }
        const int gap = int(step - hist.last);
        if(gap > 0){
            std::vector<double> rolled(hist.buf.size(), 0.0);
            const size_t age_stride = size_t(K) * 2;
            for(int age = 0; age < K; ++age){
                int src = age - gap;
                src %= K;
                if(src < 0){
                    src += K;
                }
                const double* from = hist.buf.data() + size_t(src) * age_stride;
                double* to = rolled.data() + size_t(age) * age_stride;
                for(size_t q = 0; q < age_stride; ++q){
                    to[q] = from[q];
                }
            }
            hist.buf.swap(rolled);
            for(int age = gap - 1; age >= 0; --age){
                fill_age(hist, age, rng);
            }
            hist.last = step;
        }
    }

    void zeta_of(const PairHistory& hist, double& z0, double& z1) const{
        // K(stream) times K(lag) rand -> multiply MA coefficients -> \zeta_0, \zeta_1
        // \zeta_0: smaller index particle, \zeta_1: larger index particle
            // should properly match to i,j when call
        z0 = 0.0;
        z1 = 0.0;
        const int K = m_K;
        for(int stream = 0; stream < K; ++stream){
            for(int lag = 0; lag < K; ++lag){
                const double coeff = m_A[size_t(stream) * size_t(K) + size_t(lag)];
                const size_t base = (size_t(lag) * size_t(K) + size_t(stream)) * 2;
                z0 += coeff * hist.buf[base];
                z1 += coeff * hist.buf[base + 1];
            }
        }
    }

public:
    std::vector<double> m_pair_displacement_sum;

    EffectiveRREnergyGradientAccumulator(
            std::shared_ptr<pairwise_interaction>& interaction,
            std::shared_ptr<distance_policy>& dist,
            std::vector<double> const& radii,
            double alpha, double bf, double c, unsigned long long seed)
        : m_interaction(interaction),
          m_dist(dist),
          m_coords(NULL),
          m_radii(radii),
          m_alpha(alpha),
          m_bf(bf),
          m_c(c),
          m_rho(-c),
          m_K(0),
          m_step(0),
          m_recorded_pairs(0),
          m_record_selection(false),
          m_natoms(0)
    {
        if(!(bf > 0.0 && bf <= 1.0)){
            throw std::runtime_error("effective RR: bf must lie in (0, 1]");
        }
        const int K = int(std::lround(1.0 / bf));
        if(std::abs(double(K) * bf - 1.0) > 1e-8){
            throw std::runtime_error("effective RR: bf must be 1/K for an integer K");
        }
        if(c < -1.0 || c > 0.0){
            throw std::runtime_error("effective RR: c must lie in [-1, 0]");
        }
        m_K = K;
        m_A = effective_rr_ma_coefficients(K, bf);
        const int nstripes = 64;
        m_stripes.reserve(nstripes);
        for(int s = 0; s < nstripes; ++s){
            m_stripes.push_back(std::unique_ptr<Stripe>(new Stripe()));
        }
#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
#else
        const int nthreads = 1;
#endif
        // One generator per one OpenMP thread
        m_noises.reserve(size_t(nthreads));
        for(int t = 0; t < nthreads; ++t){
            const std::uint64_t mix = 0x9E3779B97F4A7C15ULL * std::uint64_t(t + 1);
            m_noises.push_back(ThreadNoise(std::uint64_t(seed) + mix));
        }
        m_thread_grad.resize(size_t(nthreads));
        m_thread_pair_sum.resize(size_t(nthreads));
        m_thread_selection_key.resize(size_t(nthreads));
        m_thread_selection_zeta.resize(size_t(nthreads));
        m_thread_energy.assign(size_t(nthreads), 0.0);
        m_pair_displacement_sum.assign(m_ndim, 0.0);
    }

    void begin_step(const std::vector<double>* coords, size_t natoms){
        m_coords = coords;
        m_natoms = natoms;
        for(size_t s = 0; s < m_stripes.size(); ++s){
            std::lock_guard<std::mutex> lock(m_stripes[s]->mutex);
            for(auto it = m_stripes[s]->pairs.begin(); it != m_stripes[s]->pairs.end(); ){
                if(m_step - it->second.last >= m_K){
                    it = m_stripes[s]->pairs.erase(it);
                }
                else{
                    ++it;
                }
            }
        }
        for(size_t t = 0; t < m_thread_grad.size(); ++t){
            m_thread_grad[t].assign(m_ndim * natoms, 0.0);
            m_thread_pair_sum[t].assign(m_ndim, 0.0);
            m_thread_energy[t] = 0.0;
        }
    }

    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom){
        if(isubdom >= m_thread_grad.size()){
            throw std::runtime_error("effective RR: subdomain index exceeds the OpenMP thread count");
        }
        ha::VecN<m_ndim, double> dr;
        const size_t xi_off = m_ndim * atom_i;
        const size_t xj_off = m_ndim * atom_j;
        m_dist->get_rij(dr.data(), m_coords->data() + xi_off, m_coords->data() + xj_off);
        double r2 = 0.0;
        for(size_t k = 0; k < m_ndim; ++k){
            r2 += dr[k] * dr[k];
        }
        const double r = std::sqrt(r2);
        double radius_sum = 0.0;
        if(m_radii.size() > 0){
            radius_sum = m_radii[atom_i] + m_radii[atom_j];
        }
        /*********************************************************************** */
        /* Only pairs inside the interaction range (overlapped pair) are updated */
        if(!(r > 0.0 && r < radius_sum)){
            return;
        }
        /*********************************************************************** */
        double gij = 0.0;
        const double pair_energy = m_interaction->energy_gradient(r2, &gij, radius_sum);

        size_t lo = atom_i;
        size_t hi = atom_j;
        if(lo > hi){
            const size_t tmp = lo;
            lo = hi;
            hi = tmp;
        }
        const std::uint64_t key = (std::uint64_t(lo) << 32) | std::uint64_t(hi);
        Stripe& stripe = *m_stripes[size_t(key % std::uint64_t(m_stripes.size()))];
        double z_lo = 0.0;
        double z_hi = 0.0;
        {
            // map: have $K$ recent Gaussians for each pairs
            // a pair visits only one thread in a cell list
            std::lock_guard<std::mutex> lock(stripe.mutex);
            PairHistory& hist = stripe.pairs[key];
            if(hist.buf.empty()){
                hist.last = -1;
            }
            // A second visit in the same step would apply the pair force twice.
            if(hist.last == m_step){
                return;
            }
            refresh(hist, m_noises[isubdom]);
            zeta_of(hist, z_lo, z_hi);
        }
        // for autocorr and PSD calculation
        if(m_record_selection){
            m_thread_selection_key[isubdom].push_back(key);
            m_thread_selection_zeta[isubdom].push_back(z_lo);
        }
        m_thread_energy[isubdom] += pair_energy;
        const double z_on_i = (atom_i == lo) ? z_lo : z_hi;
        const double z_on_j = (atom_j == lo) ? z_lo : z_hi;
        /************************************************** */
        /**************** equation of motion ****************/
        // dx_i = -\alpha (b_f + \zeta_i) \nabla_{i} V_{ji}
        //      = \alpha (b_f + \zeta_i) g_{ij} dr
        //      (dr = x_i - x_j)
        // c = -1 => \zeta_i == \zeta_j \therefore dx_i + dx_j = 0
        for(size_t k = 0; k < m_ndim; ++k){
            const double kick_i = m_alpha * (m_bf + z_on_i) * gij * dr[k];
            const double kick_j = -m_alpha * (m_bf + z_on_j) * gij * dr[k];
            m_thread_grad[isubdom][xi_off + k] += kick_i;
            m_thread_grad[isubdom][xj_off + k] += kick_j;
            m_thread_pair_sum[isubdom][k] += kick_i + kick_j;
        }
        /************************************************** */
    }

    void finish_step(std::vector<double>& disp, double kappa){
        std::fill(disp.begin(), disp.end(), 0.0);
        m_pair_displacement_sum.assign(m_ndim, 0.0);
        for(size_t t = 0; t < m_thread_grad.size(); ++t){
            for(size_t q = 0; q < disp.size(); ++q){
                disp[q] += m_thread_grad[t][q];
            }
            for(size_t k = 0; k < m_ndim; ++k){
                m_pair_displacement_sum[k] += m_thread_pair_sum[t][k];
            }
        }
        // Harmonic potential (1/2){\kappa}{x}^{2} case (no periodic condition)
        if(kappa != 0.0 && m_coords != NULL){
            const std::vector<double>& x = *m_coords;
            for(size_t q = 0; q < disp.size(); ++q){
                disp[q] -= m_alpha * kappa * x[q];
            }
        }
        // for autocorr and PSD
            // record \zeta every step
        if(m_record_selection){
            std::vector<std::pair<std::uint64_t, double>> gathered;
            for(size_t t = 0; t < m_thread_selection_key.size(); ++t){
                for(size_t q = 0; q < m_thread_selection_key[t].size(); ++q){
                    gathered.push_back(std::make_pair(
                        m_thread_selection_key[t][q], m_thread_selection_zeta[t][q]));
                }
                m_thread_selection_key[t].clear();
                m_thread_selection_zeta[t].clear();
            }
            std::sort(gathered.begin(), gathered.end(),
                [](const std::pair<std::uint64_t, double>& a,
                   const std::pair<std::uint64_t, double>& b){ return a.first < b.first; });
            for(size_t q = 0; q < gathered.size(); ++q){
                m_selection_trace.push_back(gathered[q].second);
            }
            m_recorded_pairs = gathered.size();
        }
        m_step += 1;
    }

    void enable_selection_trace(){
        m_record_selection = true;
        m_selection_trace.clear();
        m_recorded_pairs = 0;
    }

    const std::vector<double>& selection_trace() const{
        return m_selection_trace;
    }

    double get_energy() const{
        double energy = 0.0;
        for(size_t t = 0; t < m_thread_energy.size(); ++t){
            energy += m_thread_energy[t];
        }
        return energy;
    }
};

template <typename pairwise_interaction, typename distance_policy>
class EffectiveRRCellListPotential : public CellListPotential<pairwise_interaction, distance_policy>{
// connect the accumulator to cell list
    // update -> generate zeta -> evaluate displacement -> finish_step
protected:
    EffectiveRREnergyGradientAccumulator<pairwise_interaction, distance_policy> m_acc;
    double m_kappa;
public:
    EffectiveRRCellListPotential(
            std::shared_ptr<pairwise_interaction> interaction,
            std::shared_ptr<distance_policy> dist,
            std::vector<double> const& boxvec,
            double rcut, double ncellx_scale,
            const std::vector<double> radii,
            double alpha, double bf, double c, double kappa,
            unsigned long long seed,
            const bool balance_omp=true)
        : CellListPotential<pairwise_interaction, distance_policy>(
              interaction, dist, boxvec, rcut, ncellx_scale, radii, balance_omp),
          m_acc(interaction, dist, radii, alpha, bf, c, seed),
          m_kappa(kappa)
    {}

    // Displacement happens by base_algorithm::one_step via coords += disp
    void get_displacement(std::vector<double> const& coords, std::vector<double>& disp){
        const size_t natoms = coords.size() / distance_policy::_ndim;
        if(distance_policy::_ndim * natoms != coords.size()){
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }
        if(coords.size() != disp.size()){
            throw std::invalid_argument("the gradient has the wrong size");
        }
        if(!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])){
            std::fill(disp.begin(), disp.end(), NAN);
            return;
        }
        this->update_iterator(coords);
        m_acc.begin_step(&coords, natoms);
        auto looper = this->m_cell_lists.get_atom_pair_looper(m_acc);
        looper.loop_through_atom_pairs();
        m_acc.finish_step(disp, m_kappa);
    }

    const std::vector<double>& pair_displacement_sum() const{
        return m_acc.m_pair_displacement_sum;
    }

    // for autocorr and PSD
    void enable_selection_trace(){ m_acc.enable_selection_trace(); }
    const std::vector<double>& selection_trace() const{ return m_acc.selection_trace(); }
};

} // end of namespace ha

#endif