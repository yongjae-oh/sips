#ifndef SGD_ALGORITHMS_HPP
#define SDG_ALGORITHMS_HPP
#include "utils/json.hpp"
#include "base_algorithm.hpp"
#include "potentials/inversepower_potential.hpp"

template<size_t ndim>
class inversepower_probabilistic_pairwise_sgd_clist: public base_potential_algorithm<ha::InversePowerPeriodicProbabilisticPairBatchCellLists<ndim> >{
public: 
    inversepower_probabilistic_pairwise_sgd_clist(double pow, double eps, double lr, double prob,
                  std::vector<double> const radii, 
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerPeriodicProbabilisticPairBatchCellLists<ndim> >(
                    std::make_shared<ha::InversePowerPeriodicProbabilisticPairBatchCellLists<ndim> >(pow,eps,lr,prob,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
};

template<size_t ndim>
class inversepower_probabilistic_particlewise_sgd_clist: public base_potential_algorithm<ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<ndim> >{
public: 
    inversepower_probabilistic_particlewise_sgd_clist(double pow, double eps, double lr, double prob,
                  std::vector<double> const radii, 
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<ndim> >(
                    std::make_shared<ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<ndim> >(pow,eps,lr,prob,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
};

template<size_t ndim>
class inversepower_correlated_probabilistic_pairwise_sgd_clist: public base_potential_algorithm<ha::InversePowerPeriodicCorrelatedProbabilisticPairBatchCellLists<ndim> >{
public: 
    inversepower_correlated_probabilistic_pairwise_sgd_clist(double pow, double eps, double lr, double prob, double correlation,
                  std::vector<double> const radii, 
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerPeriodicCorrelatedProbabilisticPairBatchCellLists<ndim> >(
                    std::make_shared<ha::InversePowerPeriodicCorrelatedProbabilisticPairBatchCellLists<ndim> >(pow,eps,lr,prob,correlation,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
};

// periodic boundary algorithm of the effective SDE of RR-SGD
    // 261113 YO
    // if base_algorithm::one_step sends a particle outside the box, move it properly into the box
template<size_t ndim>
class inversepower_effective_rr_periodic_clist: public base_potential_algorithm<ha::InversePowerPeriodicEffectiveRRCellLists<ndim> >{
public:
    inversepower_effective_rr_periodic_clist(double pow, double eps, double alpha, double bf, double c, double kappa,
                  unsigned long long seed,
                  std::vector<double> const radii,
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerPeriodicEffectiveRRCellLists<ndim> >(
                    std::make_shared<ha::InversePowerPeriodicEffectiveRRCellLists<ndim> >(pow,eps,alpha,bf,c,kappa,seed,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
    virtual void one_step(){
        base_algorithm<ha::InversePowerPeriodicEffectiveRRCellLists<ndim> >::one_step();
        this->m_potential->wrap(this->m_coords);
    }
    std::vector<double> coordinates() const { return this->m_coords; }
    double energy(){ return (this->m_potential)->get_energy(this->m_coords); }
    std::vector<double> pair_displacement_sum() const { return (this->m_potential)->pair_displacement_sum(); }
        // for autocorr and PSD
    void enable_selection_trace(){ (this->m_potential)->enable_selection_trace(); }
    std::vector<double> selection_trace() const { return (this->m_potential)->selection_trace(); }
};

    // same for harmonic case (no periodic)
template<size_t ndim>
class inversepower_effective_rr_harmonic_clist: public base_potential_algorithm<ha::InversePowerCartesianEffectiveRRCellLists<ndim> >{
public:
    inversepower_effective_rr_harmonic_clist(double pow, double eps, double alpha, double bf, double c, double kappa,
                  unsigned long long seed,
                  std::vector<double> const radii,
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerCartesianEffectiveRRCellLists<ndim> >(
                    std::make_shared<ha::InversePowerCartesianEffectiveRRCellLists<ndim> >(pow,eps,alpha,bf,c,kappa,seed,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
    std::vector<double> coordinates() const { return this->m_coords; }
    double energy(){ return (this->m_potential)->get_energy(this->m_coords); }
    std::vector<double> pair_displacement_sum() const { return (this->m_potential)->pair_displacement_sum(); }
        // for autocorr and PSD
    void enable_selection_trace(){ (this->m_potential)->enable_selection_trace(); }
    std::vector<double> selection_trace() const { return (this->m_potential)->selection_trace(); }
};

// periodic boundary algorithm of the  RR-SGD
    // 261113 YO
    // if base_algorithm::one_step sends a particle outside the box, move it properly into the box
template<size_t ndim>
class inversepower_rr_sgd_periodic_clist: public base_potential_algorithm<ha::InversePowerPeriodicRRSGDCellLists<ndim> >{
public:
    inversepower_rr_sgd_periodic_clist(double pow, double eps, double alpha, double bf, double c, double kappa,
                  unsigned long long seed,
                  std::vector<double> const radii,
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerPeriodicRRSGDCellLists<ndim> >(
                    std::make_shared<ha::InversePowerPeriodicRRSGDCellLists<ndim> >(pow,eps,alpha,bf,c,kappa,seed,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
    virtual void one_step(){
        base_algorithm<ha::InversePowerPeriodicRRSGDCellLists<ndim> >::one_step();
        this->m_potential->wrap(this->m_coords);
    }
    std::vector<double> coordinates() const { return this->m_coords; }
    double energy(){ return (this->m_potential)->get_energy(this->m_coords); }
    std::vector<double> pair_displacement_sum() const { return (this->m_potential)->pair_displacement_sum(); }
        // for autocorr and PSD
    void enable_selection_trace(){ (this->m_potential)->enable_selection_trace(); }
    std::vector<double> selection_trace() const { return (this->m_potential)->selection_trace(); }
};


#endif
