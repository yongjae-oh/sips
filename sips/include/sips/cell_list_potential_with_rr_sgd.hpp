#ifndef SIPS_CELL_LIST_POTENTIAL_WITH_PROBABILISTIC_BATCH_HPP
#define SIPS_CELL_LIST_POTENTIAL_WITH_PROBABILISTIC_BATCH_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <omp.h>
//#include <math.h>
#include <cmath>
#include <random>
#include <set>

#include <cstdint>
#include <stdexcept>

#include "hyperalg/distance.hpp"
#include "hyperalg/cell_list_potential.hpp"
#include "hyperalg/vecN.hpp"
#include "utils/noise.hpp"

namespace ha{

/* RR-SGD (SGD without replacement, or SGD with random reshuffling) */
// K = K_max (concept of inclusion gets trivial -> to be extended later)
// slot is assigned to pairs before the overlap is evaluated
// c \in [-1, 0]
    // if c = 0: ij and ji use independent slots
    // if not: ij and ji shares the slot with probability $c$

// equation of motion: \Delta[x}_{i} = -\alpha \theta_{ji} \nabla_{i} V_{ji}

// hash based mixer (random rearranger, effective integer mixer)
inline std::uint64_t rr_sgd_mix(std::uint64_t x){
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

template <typename pairwise_interaction, typename distance_policy>
class RRSGDEnergyGradientAccumulator{
protected:
    const static size_t m_ndim = distance_policy::_ndim;
    std::shared_ptr<pairwise_interaction> m_interaction;
    std::shared_ptr<distance_policy> m_dist;
    const std::vector<double>* m_coords;
    std::vector<double> m_radii;
    double m_alpha;
    double m_bf;
    double m_c;
    int m_K;
    std::uint64_t m_seed;
    long long m_step;
    size_t m_natoms;
    bool m_record_selection;
    std::vector<double> m_selection_trace;
    std::vector<std::vector<double>> m_thread_grad;
    std::vector<std::vector<double>> m_thread_pair_sum;

    void slots_of(size_t lo, size_t hi, int epoch, int& slot_lo, int& slot_hi) const{

        // assigning slots
            // particle index: lo (low), hi (high), selected not knowing whether the pairs are overlapped
            // first sample u~Unif(0, 1) for hi
            // if K == 1 or c == -1 hi-lo and lo-hi share a slot
            // else if u < ((-1) * c), hi-lo and lo-hi share a slot
            // else, secondly sample another slot with new hash
                // ==> corr(\theta_{ji},\,\theta_{ij}) = -c.
        const std::uint64_t h = rr_sgd_mix(
            m_seed
            ^ (std::uint64_t(lo) + 0x9E3779B97F4A7C15ULL)
            ^ (std::uint64_t(hi) << 32)
            ^ (std::uint64_t(epoch) * 0xD1B54A32D192ED03ULL));
        // u is uniform on [0, 1). Share a slot with probability -c.
        const double u = double(h >> 11) * (1.0 / 9007199254740992.0);
        const bool share = (m_K == 1) || (u < -m_c);
        if(share){
            const int slot = int(h % std::uint64_t(m_K));
            slot_lo = slot;
            slot_hi = slot;
            return;
        }
        const std::uint64_t h2 = rr_sgd_mix(h ^ 0xD1B54A32D192ED03ULL);
        slot_lo = int(h % std::uint64_t(m_K));
        slot_hi = int(h2 % std::uint64_t(m_K));
    }

public:
    std::vector<double> m_pair_displacement_sum;

    RRSGDEnergyGradientAccumulator(
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
          m_K(0),
          m_seed(std::uint64_t(seed)),
          m_step(0),
          m_natoms(0),
          m_record_selection(false)
    {
        if(!(bf > 0.0 && bf <= 1.0)){
            throw std::runtime_error("RR SGD: bf must lie in (0, 1]");
        }
        const int K = int(std::lround(1.0 / bf));
        if(std::abs(double(K) * bf - 1.0) > 1e-8){
            throw std::runtime_error("RR SGD: bf must be 1/K for an integer K");
        }
        if(c < -1.0 || c > 0.0){
            throw std::runtime_error("RR SGD: c must lie in [-1, 0]");
        }
        m_K = K;
#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
#else
        const int nthreads = 1;
#endif
        m_thread_grad.resize(size_t(nthreads));
        m_thread_pair_sum.resize(size_t(nthreads));
        m_pair_displacement_sum.assign(m_ndim, 0.0);
    }

    void begin_step(const std::vector<double>* coords, size_t natoms){
        m_coords = coords;
        m_natoms = natoms;
        for(size_t t = 0; t < m_thread_grad.size(); ++t){
            m_thread_grad[t].assign(m_ndim * natoms, 0.0);
            m_thread_pair_sum[t].assign(m_ndim, 0.0);
        }
    }

    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom){
        if(isubdom >= m_thread_grad.size()){
            throw std::runtime_error("RR SGD: subdomain index exceeds the OpenMP thread count");
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
        if(!(r > 0.0 && r < radius_sum)){
            return;
        }
        double gij = 0.0;
        m_interaction->energy_gradient(r2, &gij, radius_sum);
        size_t lo = atom_i;
        size_t hi = atom_j;
        if(lo > hi){
            const size_t tmp = lo;
            lo = hi;
            hi = tmp;
        }
        int slot_lo = 0;
        int slot_hi = 0;
        slots_of(lo, hi, int(m_step / m_K), slot_lo, slot_hi);
        const int phase = int(m_step % m_K);
        /**********************************************************/
        // \theta becomes 1 only when the phase matches with the slot
        const double theta_lo = (slot_lo == phase) ? 1.0 : 0.0;
        const double theta_hi = (slot_hi == phase) ? 1.0 : 0.0;
        const double theta_i = (atom_i == lo) ? theta_lo : theta_hi;
        const double theta_j = (atom_j == lo) ? theta_lo : theta_hi;
        
        // no update when \theta_{i} and \theta_{j} are both zero
        if(theta_i == 0.0 && theta_j == 0.0){
            return;
        }
        
        // calulate kick with previously decided \theta, and then update
        for(size_t k = 0; k < m_ndim; ++k){
            const double kick_i = m_alpha * theta_i * gij * dr[k];
            const double kick_j = -m_alpha * theta_j * gij * dr[k];
            m_thread_grad[isubdom][xi_off + k] += kick_i;
            m_thread_grad[isubdom][xj_off + k] += kick_j;
            m_thread_pair_sum[isubdom][k] += kick_i + kick_j;
        }
        /**********************************************************/
    }

    void finish_step(std::vector<double>& disp, double kappa){
        // aggregate displacements of each thread
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
            
        // harmonic potential case, open (not periodic) boundary condition
            // add the elastic central force -\alpha{\kappa}{x}
        if(kappa != 0.0 && m_coords != NULL){
            const std::vector<double>& x = *m_coords;
            for(size_t q = 0; q < disp.size(); ++q){
                disp[q] -= m_alpha * kappa * x[q];
            }
        }
        
        // for autocorr and PSD
        if(m_record_selection){
            const int epoch = int(m_step / m_K);
            const int phase = int(m_step % m_K);
            for(size_t lo = 0; lo < m_natoms; ++lo){
                for(size_t hi = lo + 1; hi < m_natoms; ++hi){
                    int slot_lo = 0;
                    int slot_hi = 0;
                    slots_of(lo, hi, epoch, slot_lo, slot_hi);
                    const double theta_lo = (slot_lo == phase) ? 1.0 : 0.0;
                    m_selection_trace.push_back(theta_lo);
                    (void)slot_hi;
                }
            }
        }
        m_step += 1;
    }

    void enable_selection_trace(){
        m_record_selection = true;
        m_selection_trace.clear();
    }

    const std::vector<double>& selection_trace() const{
        return m_selection_trace;
    }
};

template <typename pairwise_interaction, typename distance_policy>
class RRSGDCellListPotential : public CellListPotential<pairwise_interaction, distance_policy>{
// connect the accumulator to cell list
    // evaluate neighbor -> update overlapped pairs -> finish_step
protected:
    RRSGDEnergyGradientAccumulator<pairwise_interaction, distance_policy> m_acc;
    double m_kappa;
public:
    RRSGDCellListPotential(
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


} // namespace ha ends

#endif //#ifndef 
